"""Smoke tests for Struct2SeqAO and AO-ARM ELBO (Eq. 10)."""
from __future__ import annotations

import sys
import numpy as np
import torch

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from struct2seq.struct2seq_ao import Struct2SeqAO


def _make_model_and_data(
    B: int = 2,
    N: int = 20,
) -> tuple[Struct2SeqAO, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    device = torch.device("cpu")
    model = Struct2SeqAO(
        num_letters=20,
        node_features=64,
        edge_features=64,
        hidden_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        vocab=20,
        k_neighbors=10,
        dropout=0.0,
    ).to(device)
    X = torch.randn(B, N, 4, 3, device=device)
    S = torch.randint(0, 20, (B, N), device=device)
    lengths = np.array([N, N - 3])
    mask = torch.zeros(B, N, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return model, X, S, lengths, mask


def test_basic() -> None:
    """Loss is finite, scalar, and gradients flow into encoder / decoder."""
    torch.manual_seed(42)
    model, X, S, lengths, mask = _make_model_and_data()

    loss, info = model.compute_elbo_ao(X, S, lengths, mask)

    assert loss.requires_grad, "loss must require grad"
    assert loss.shape == (), "loss must be scalar"
    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
    assert "elbo" in info and "nll" in info

    loss.backward()

    # Check that encoder / decoder parameters receive gradients
    grad_total = 0.0
    for name, p in model.named_parameters():
        if any(tag in name for tag in ("W_v", "W_e", "W_s", "encoder_layers", "decoder_layers", "W_out")):
            if p.grad is not None:
                grad_total += float(p.grad.abs().sum().item())
    assert grad_total > 0.0, "encoder/decoder parameters must receive non-zero gradients"

    print("test_basic PASSED")
    print(f"  loss  = {loss.item():.4f}")
    print(f"  ELBO  = {info['elbo'].item():.4f}")
    print(f"  NLL   = {info['nll'].item():.4f}")
    print(f"  i_mean= {info['i_mean'].item():.1f}")


def test_edge_i_equals_1() -> None:
    """When i=1, remaining_mask should equal mask and F reduces to sum over all sites."""
    torch.manual_seed(0)
    model, X, S, lengths, mask = _make_model_and_data()

    B, N = S.shape
    device = S.device

    h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)

    # Force i=1 for all batch elements (no decoded positions)
    L_tensor = torch.tensor(lengths, dtype=torch.float, device=device)
    i_samples = torch.ones(B, dtype=torch.long, device=device)

    # Random permutation per batch, restricted to valid positions [0, L_b)
    perms = []
    for b in range(B):
        L_b = int(lengths[b])
        valid = torch.arange(N, device=device)[:L_b]
        invalid = torch.arange(N, device=device)[L_b:]
        perm_valid = valid[torch.randperm(L_b, device=device)]
        full_b = torch.cat([perm_valid, invalid], dim=0)
        perms.append(full_b)
    full_perm = torch.stack(perms, dim=0)

    ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_samples)
    log_probs, _ = model.forward_p(h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask)

    log_p_token = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)

    rank = torch.zeros(B, N, dtype=torch.long, device=device)
    rank.scatter_(
        1,
        full_perm,
        torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
    )
    decoded_mask = (rank < (i_samples - 1).unsqueeze(1)).float()

    remaining_mask = (1.0 - decoded_mask) * mask
    # For i=1, no decoded positions
    assert torch.all(decoded_mask == 0), "decoded_mask must be all-zero when i=1"
    assert torch.all(remaining_mask == mask), "remaining_mask must equal mask when i=1"

    sum_rem = (log_p_token * remaining_mask).sum(-1)
    cnt_rem = remaining_mask.sum(-1)
    cnt_rem = cnt_rem.clamp(min=1.0)

    F_impl = (L_tensor / cnt_rem) * sum_rem
    F_expected = (log_p_token * mask).sum(-1)

    assert torch.allclose(F_impl, F_expected, atol=1e-5), "F(i=1) mismatch"
    print(f"test_edge_i_equals_1 PASSED  F={F_impl.tolist()}")


def test_edge_i_equals_L() -> None:
    """When i=L, exactly one position remains and F = L * log p at that site."""
    torch.manual_seed(1)
    model, X, S, lengths, mask = _make_model_and_data()

    B, N = S.shape
    device = S.device

    h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)

    L_tensor = torch.tensor(lengths, dtype=torch.float, device=device)
    i_samples = L_tensor.long()

    perms = []
    for b in range(B):
        L_b = int(lengths[b])
        valid = torch.arange(N, device=device)[:L_b]
        invalid = torch.arange(N, device=device)[L_b:]
        perm_valid = valid[torch.randperm(L_b, device=device)]
        full_b = torch.cat([perm_valid, invalid], dim=0)
        perms.append(full_b)
    full_perm = torch.stack(perms, dim=0)

    rank = torch.zeros(B, N, dtype=torch.long, device=device)
    rank.scatter_(
        1,
        full_perm,
        torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
    )
    decoded_mask = (rank < (i_samples - 1).unsqueeze(1)).float()

    remaining_mask = (1.0 - decoded_mask) * mask
    remaining_count = remaining_mask.sum(-1)
    assert torch.all(remaining_count == 1), (
        f"Exactly 1 position should remain, got {remaining_count.tolist()}"
    )

    ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_samples)
    log_probs, _ = model.forward_p(h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask)
    log_p_token = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)

    sum_rem = (log_p_token * remaining_mask).sum(-1)
    cnt_rem = remaining_mask.sum(-1)
    cnt_rem = cnt_rem.clamp(min=1.0)
    F_impl = (L_tensor / cnt_rem) * sum_rem

    # For each batch, identify the remaining index and check formula
    remaining_idx = remaining_mask.argmax(dim=-1)
    log_p_remaining = log_p_token.gather(1, remaining_idx.unsqueeze(1)).squeeze(1)
    F_expected = L_tensor * log_p_remaining

    assert torch.allclose(F_impl, F_expected, atol=1e-5), "F(i=L) mismatch"
    print(f"test_edge_i_equals_L PASSED  F={F_impl.tolist()}")


if __name__ == "__main__":
    test_basic()
    test_edge_i_equals_1()
    test_edge_i_equals_L()
    print("\nALL TESTS PASSED")

