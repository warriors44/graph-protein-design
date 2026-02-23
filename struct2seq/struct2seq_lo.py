from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import (
    TransformerLayer,
    MPNNLayer,
    cat_neighbors_nodes,
    gather_nodes,
)
from .protein_features import ProteinFeatures
from .gumbel import gumbel_top_k, plackett_luce_log_prob


class Struct2SeqLO(nn.Module):
    """Struct2Seq with Learning-Order autoregressive decoding.

    Extends the original Struct2Seq architecture with a learned decoding order
    based on the Plackett-Luce distribution and Gumbel Top-K sampling,
    following the framework of Wang et al. (arXiv:2503.05979).

    The model has three probability components:
      - p_theta(x_{z_i} | x_{z_{<i}}, s): token prediction (shared decoder)
      - p_theta(z_i | z_{<i}, x_{z_{<i}}, s): order prior (W_order_p head)
      - q_theta(z_i | z_{<i}, x, s): variational order posterior (W_order_q head)
    """

    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 20,
        k_neighbors: int = 30,
        protein_features: str = 'full',
        augment_eps: float = 0.,
        dropout: float = 0.1,
        forward_attention_decoder: bool = True,
        use_mpnn: bool = False,
    ) -> None:
        super(Struct2SeqLO, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout,
        )

        # Embedding layers (shared)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Shared encoder
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Shared decoder
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Token prediction head (shared)
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Order prediction heads
        self.W_order_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.W_order_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Encoder (shared between p and q paths)
    # ------------------------------------------------------------------

    def _encode(
        self,
        X: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run structure encoder.

        Returns:
            h_V: encoded node features [B, N, H]
            h_E: encoded edge features [B, N, K, H]
            E_idx: neighbor indices [B, N, K]
            mask_attend_enc: encoder attention mask [B, N, K]
        """
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for enc_layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = enc_layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return h_V, h_E, E_idx, mask_attend

    # ------------------------------------------------------------------
    # Generalized autoregressive mask
    # ------------------------------------------------------------------

    def _generalized_autoregressive_mask(
        self,
        E_idx: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        """Build autoregressive mask from an arbitrary decoding permutation.

        Args:
            E_idx: neighbor indices [B, N, K].
            permutation: [B, N] where permutation[b, i] = index of position
                decoded at step i.

        Returns:
            mask: [B, N, K] with 1 where neighbor was decoded earlier in
                the permutation (i.e. is in the "past").
        """
        B, N = permutation.shape
        device = permutation.device

        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(1, permutation, torch.arange(N, device=device).unsqueeze(0).expand(B, -1))

        rank_self = rank.unsqueeze(-1)

        rank_flat = rank.unsqueeze(1).expand(-1, N, -1)
        rank_neighbors = torch.gather(rank_flat, 2, E_idx)

        mask = (rank_neighbors < rank_self).float()
        return mask

    # ------------------------------------------------------------------
    # q_theta path: unmasked decoder for variational posterior
    # ------------------------------------------------------------------

    def forward_q(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q_theta order logits using the unmasked (full info) decoder.

        All sequence information is visible (no causal mask).

        Args:
            h_V_enc: encoder output [B, N, H].
            h_E: edge embeddings [B, N, K, H].
            E_idx: neighbor indices [B, N, K].
            S: ground-truth sequence [B, N].
            mask: padding mask [B, N].

        Returns:
            q_logits: [B, N] per-position order logits.
        """
        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for dec_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = dec_layer(h_V, h_ESV, mask_V=mask, mask_attend=mask_attend)

        q_logits = self.W_order_q(h_V).squeeze(-1)
        q_logits = q_logits.masked_fill(mask == 0, float('-inf'))
        return q_logits

    # ------------------------------------------------------------------
    # p_theta path: generalized causal decoder
    # ------------------------------------------------------------------

    def forward_p(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        permutation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run decoder with a generalized autoregressive mask defined by
        the given permutation.

        Args:
            h_V_enc: encoder output [B, N, H].
            h_E: edge embeddings [B, N, K, H].
            E_idx: neighbor indices [B, N, K].
            S: ground-truth sequence [B, N] (teacher forcing).
            mask: padding mask [B, N].
            permutation: decoding order [B, N].

        Returns:
            log_probs: [B, N, vocab] log-probabilities for each position.
            p_order_logits: [B, N] per-position order logits for p_theta.
        """
        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        ar_mask = self._generalized_autoregressive_mask(E_idx, permutation)
        mask_attend = ar_mask.unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend

        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0

        for dec_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = dec_layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        p_order_logits = self.W_order_p(h_V).squeeze(-1)
        p_order_logits = p_order_logits.masked_fill(mask == 0, float('-inf'))

        return log_probs, p_order_logits

    # ------------------------------------------------------------------
    # ELBO computation
    # ------------------------------------------------------------------

    def compute_elbo(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the variational ELBO for learning-order training.

        F_theta = sum_i [ log p(x_{z_i} | x_{z_{<i}}, s)
                        + log p(z_i | z_{<i}, x_{z_{<i}}, s)
                        - log q(z_i | z_{<i}, x, s) ]

        Args:
            X: coordinates [B, N, 4, 3].
            S: ground-truth sequence [B, N].
            L: lengths array.
            mask: padding mask [B, N].

        Returns:
            elbo: scalar ELBO (averaged over valid residues).
            info: dict with intermediate values for logging.
        """
        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        # --- q_theta path ---
        q_logits = self.forward_q(h_V_enc, h_E, E_idx, S, mask)
        permutation = gumbel_top_k(q_logits)

        # --- p_theta path ---
        log_probs, p_order_logits = self.forward_p(
            h_V_enc, h_E, E_idx, S, mask, permutation,
        )

        # --- Token NLL: log p(x_{z_i} | x_{z_{<i}}, s) ---
        log_p_token = torch.gather(
            log_probs, 2, S.unsqueeze(-1),
        ).squeeze(-1)

        # --- Order log-probs under Plackett-Luce ---
        log_q_order = plackett_luce_log_prob(q_logits, permutation, mask)
        log_p_order = plackett_luce_log_prob(p_order_logits, permutation, mask)

        # --- F_theta per position (in permutation order) ---
        B, N = S.shape
        rank = torch.zeros(B, N, dtype=torch.long, device=S.device)
        rank.scatter_(1, permutation, torch.arange(N, device=S.device).unsqueeze(0).expand(B, -1))

        log_p_token_perm = torch.gather(log_p_token, 1, permutation)
        mask_perm = torch.gather(mask, 1, permutation)

        f_per_step = (
            log_p_token_perm
            + log_p_order
            - log_q_order
        ) * mask_perm

        elbo = f_per_step.sum(-1)
        elbo_avg = elbo.sum() / mask.sum()

        nll_avg = -(log_p_token * mask).sum() / mask.sum()

        info = {
            'elbo': elbo_avg.detach(),
            'nll': nll_avg.detach(),
            'log_p_token': (log_p_token_perm * mask_perm).sum().detach() / mask.sum(),
            'log_p_order': (log_p_order * mask_perm).sum().detach() / mask.sum(),
            'log_q_order': (log_q_order * mask_perm).sum().detach() / mask.sum(),
            'permutation': permutation.detach(),
        }

        return elbo_avg, info

    # ------------------------------------------------------------------
    # Forward (convenience wrapper matching original Struct2Seq interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward pass using sequential (N->C) ordering.

        Compatible with the original Struct2Seq interface for evaluation.
        """
        B, N = S.shape
        device = S.device
        permutation = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)
        log_probs, _ = self.forward_p(h_V_enc, h_E, E_idx, S, mask, permutation)
        return log_probs

    # ------------------------------------------------------------------
    # Sampling with learned order
    # ------------------------------------------------------------------

    def sample(
        self,
        X: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling using the learned order (p_theta only).

        At each step:
          1. Compute order logits for remaining positions -> select next position.
          2. Compute amino acid logits for the selected position -> sample token.

        Args:
            X: coordinates [B, N, 4, 3].
            L: lengths array.
            mask: padding mask [B, N].
            temperature: sampling temperature.

        Returns:
            S: sampled sequence [B, N].
            ordering: [B, N] the order in which positions were decoded.
        """
        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        B, N_nodes = X.size(0), X.size(1)
        device = X.device

        S = torch.zeros(B, N_nodes, dtype=torch.long, device=device)
        ordering = torch.zeros(B, N_nodes, dtype=torch.long, device=device)
        decoded_mask = torch.zeros(B, N_nodes, device=device)
        h_S = torch.zeros_like(h_V_enc)

        h_V_stack = [h_V_enc.clone()] + [
            torch.zeros_like(h_V_enc) for _ in range(len(self.decoder_layers))
        ]

        for t in range(N_nodes):
            remaining = (mask - decoded_mask)

            if remaining.sum() == 0:
                break

            # --- Select next position via p_theta order head ---
            h_V_current = h_V_stack[-1]
            order_logits = self.W_order_p(h_V_current).squeeze(-1)
            order_logits = order_logits.masked_fill(remaining == 0, float('-inf'))
            order_logits = order_logits / temperature

            order_probs = F.softmax(order_logits, dim=-1)
            pos = torch.multinomial(order_probs, 1).squeeze(-1)
            ordering[:, t] = pos

            # --- Decode amino acid at the selected position ---
            for b in range(B):
                p = pos[b].item()
                E_idx_p = E_idx[b:b+1, p:p+1, :]
                h_E_p = h_E[b:b+1, p:p+1, :, :]
                h_ES_p = cat_neighbors_nodes(
                    h_S[b:b+1], h_E_p, E_idx_p,
                )

                ar_mask_p = decoded_mask[b:b+1].unsqueeze(0)
                neighbor_decoded = torch.gather(
                    decoded_mask[b:b+1].unsqueeze(1).expand(-1, 1, E_idx_p.size(-1)),
                    -1,
                    E_idx_p[:, 0:1, :].squeeze(1).unsqueeze(0),
                ).unsqueeze(-1)
                mask_bw_p = neighbor_decoded
                mask_fw_p = 1.0 - neighbor_decoded

                h_ES_encoder_p = cat_neighbors_nodes(
                    torch.zeros_like(h_S[b:b+1]), h_E_p, E_idx_p,
                )
                h_ESV_encoder_p = cat_neighbors_nodes(
                    h_V_enc[b:b+1], h_ES_encoder_p, E_idx_p,
                )
                h_ESV_encoder_fw_p = mask_fw_p * h_ESV_encoder_p

                for l, dec_layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_p = cat_neighbors_nodes(
                        h_V_stack[l][b:b+1], h_ES_p, E_idx_p,
                    )
                    h_V_p = h_V_stack[l][b:b+1, p:p+1, :]
                    h_ESV_p = mask_bw_p * h_ESV_decoder_p + h_ESV_encoder_fw_p
                    h_V_stack[l + 1][b, p, :] = dec_layer(
                        h_V_p, h_ESV_p, mask_V=mask[b:b+1, p:p+1],
                    ).squeeze(0).squeeze(0)

                h_V_final = h_V_stack[-1][b, p, :]
                token_logits = self.W_out(h_V_final) / temperature
                token_probs = F.softmax(token_logits, dim=-1)
                S_p = torch.multinomial(token_probs.unsqueeze(0), 1).squeeze(-1).squeeze(0)

                S[b, p] = S_p
                h_S[b, p, :] = self.W_s(S_p)

            decoded_mask.scatter_(1, pos.unsqueeze(-1), 1.0)

        return S, ordering
