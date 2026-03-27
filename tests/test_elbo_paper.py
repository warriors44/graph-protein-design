"""Smoke test for compute_elbo_paper (Algorithm 1, Eqs. 8/9/11)."""
from __future__ import annotations

import sys
import numpy as np
import torch

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from struct2seq.struct2seq_lo import Struct2SeqLO


def _make_model_and_data(
    B: int = 2,
    N: int = 20,
    separate_decoder: bool = False,
    separate_encoder: bool = False,
) -> tuple[Struct2SeqLO, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    device = torch.device('cpu')
    model = Struct2SeqLO(
        num_letters=20,
        node_features=64,
        edge_features=64,
        hidden_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        vocab=20,
        k_neighbors=10,
        dropout=0.0,
        separate_decoder=separate_decoder,
        separate_encoder=separate_encoder,
    ).to(device)
    X = torch.randn(B, N, 4, 3, device=device)
    S = torch.randint(0, 20, (B, N), device=device)
    lengths = np.array([N, N - 3])
    mask = torch.zeros(B, N, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return model, X, S, lengths, mask


def test_basic() -> None:
    """Loss is finite, requires grad, and both order heads receive gradients.

    Runs for both shared and separate decoder to ensure the ELBO
    implementation supports both parameterisations.
    """
    for separate_decoder in (False, True):
        torch.manual_seed(42)
        model, X, S, lengths, mask = _make_model_and_data(
            separate_decoder=separate_decoder,
        )

        loss, info = model.compute_elbo_paper(X, S, lengths, mask)

        assert loss.requires_grad, "loss must require grad"
        assert loss.shape == (), "loss must be scalar"
        assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
        assert 'elbo' in info and 'nll' in info

        loss.backward()

        for tag in ('W_order_q', 'W_order_p'):
            total = sum(
                p.grad.abs().sum().item()
                for n, p in model.named_parameters()
                if tag in n and p.grad is not None
            )
            assert total > 0, (
                f"{tag} must receive non-zero gradients "
                f"(separate_decoder={separate_decoder})"
            )

        label = f"separate_decoder={separate_decoder}"
        print(f"test_basic PASSED ({label})")
        print(f"  loss  = {loss.item():.4f}")
        print(f"  ELBO  = {info['elbo'].item():.4f}")
        print(f"  NLL   = {info['nll'].item():.4f}")
        print(f"  |dF|  = {info['delta_F_abs'].item():.6f}")
        print(f"  i_mean= {info['i_mean'].item():.1f}")

        q_norm = sum(
            p.grad.norm().item()
            for n, p in model.named_parameters()
            if 'W_order_q' in n and p.grad is not None
        )
        p_norm = sum(
            p.grad.norm().item()
            for n, p in model.named_parameters()
            if 'W_order_p' in n and p.grad is not None
        )
        print(f"  q_grad_norm = {q_norm:.6f}")
        print(f"  p_grad_norm = {p_norm:.6f}")


def test_entropy_bonus_keys_and_loss_consistency() -> None:
    """Entropy bonus: info keys and loss = elbo_no_penalty - lambda * entropy_q."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = Struct2SeqLO(
        num_letters=20,
        node_features=64,
        edge_features=64,
        hidden_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        vocab=20,
        k_neighbors=10,
        dropout=0.0,
        lambda_entropy=0.01,
    ).to(device)
    B, N = 2, 20
    X = torch.randn(B, N, 4, 3, device=device)
    S = torch.randint(0, 20, (B, N), device=device)
    lengths = np.array([N, N - 3])
    mask = torch.zeros(B, N, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    loss, info = model.compute_elbo_paper(X, S, lengths, mask)

    assert "entropy_q" in info
    assert "entropy_q_weighted" in info
    assert "elbo_no_penalty" in info
    assert torch.isfinite(info["entropy_q"])
    assert torch.isfinite(info["entropy_q_weighted"])
    assert torch.isfinite(info["elbo_no_penalty"])

    expected = info["elbo_no_penalty"] - 0.01 * info["entropy_q"]
    assert torch.allclose(loss.detach(), expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(
        info["entropy_q_weighted"],
        0.01 * info["entropy_q"],
        rtol=1e-5,
        atol=1e-6,
    )
    print("test_entropy_bonus_keys_and_loss_consistency PASSED")


def test_loglik_is_q_basic() -> None:
    """Basic sanity check for compute_loglik_is_q.

    Ensures that the importance-sampling estimator returns finite
    per-residue log-likelihoods for a small random problem.
    """
    torch.manual_seed(0)
    model, X, S, lengths, mask = _make_model_and_data()

    loglik_per_res = model.compute_loglik_is_q(
        X,
        S,
        lengths,
        mask,
        num_samples_eval=2,
    )

    assert loglik_per_res.shape == (S.size(0),)
    assert torch.isfinite(loglik_per_res).all(), (
        f"loglik_per_res must be finite, got {loglik_per_res}"
    )


def test_loglik_mc_p_basic() -> None:
    """Basic sanity check for compute_loglik_mc_p.

    Ensures that the p-theta Monte Carlo estimator returns finite
    per-residue log-likelihoods for a small random problem.
    """
    torch.manual_seed(0)
    model, X, S, lengths, mask = _make_model_and_data()

    loglik_per_res = model.compute_loglik_mc_p(
        X,
        S,
        lengths,
        mask,
        num_samples_eval=2,
    )

    assert loglik_per_res.shape == (S.size(0),)
    assert torch.isfinite(loglik_per_res).all(), (
        f"loglik_per_res must be finite, got {loglik_per_res}"
    )


def test_loglik_proxy_q_px_basic() -> None:
    """Basic sanity check for compute_loglik_proxy_q_px.

    Ensures that the lightweight q-sampled proxy estimator returns finite
    per-residue log-likelihoods for a small random problem.
    """
    torch.manual_seed(0)
    model, X, S, lengths, mask = _make_model_and_data()

    loglik_per_res = model.compute_loglik_proxy_q_px(
        X,
        S,
        lengths,
        mask,
        num_samples_eval=2,
    )

    assert loglik_per_res.shape == (S.size(0),)
    assert torch.isfinite(loglik_per_res).all(), (
        f"loglik_per_res must be finite, got {loglik_per_res}"
    )


def test_edge_i_equals_1() -> None:
    """When i=1 for all batch elements, no positions are decoded yet."""
    torch.manual_seed(0)
    model, X, S, lengths, mask = _make_model_and_data()

    B, N = S.shape
    device = S.device

    h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)
    q_logits = model.forward_q(h_V_enc, h_E, E_idx, S, mask)

    from struct2seq.gumbel import gumbel_top_k
    full_perm = gumbel_top_k(q_logits.detach())
    i_samples = torch.ones(B, dtype=torch.long, device=device)

    ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_samples)
    log_probs, p_order_logits = model.forward_p(
        h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask,
    )
    remaining_mask = mask.clone()
    F_val = Struct2SeqLO._compute_F_theta(
        log_probs, p_order_logits, q_logits, S, remaining_mask,
    )
    assert torch.isfinite(F_val).all(), f"F must be finite, got {F_val}"
    print(f"test_edge_i_equals_1 PASSED  F={F_val.tolist()}")


def test_edge_i_equals_L() -> None:
    """When i=L, only one position remains undecoded."""
    torch.manual_seed(1)
    model, X, S, lengths, mask = _make_model_and_data()

    B, N = S.shape
    device = S.device

    h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)
    q_logits = model.forward_q(h_V_enc, h_E, E_idx, S, mask)

    from struct2seq.gumbel import gumbel_top_k
    full_perm = gumbel_top_k(q_logits.detach())

    L_tensor = torch.tensor(lengths, device=device)
    i_samples = L_tensor.long()

    rank = torch.zeros(B, N, dtype=torch.long, device=device)
    rank.scatter_(
        1, full_perm,
        torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
    )
    decoded_mask = (rank < (i_samples - 1).unsqueeze(1)).float()

    remaining_mask = (1.0 - decoded_mask) * mask
    remaining_count = remaining_mask.sum(-1)
    assert (remaining_count == 1).all(), (
        f"Exactly 1 position should remain, got {remaining_count.tolist()}"
    )

    ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_samples)
    log_probs, p_order_logits = model.forward_p(
        h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask,
    )
    F_val = Struct2SeqLO._compute_F_theta(
        log_probs, p_order_logits, q_logits, S, remaining_mask,
    )
    assert torch.isfinite(F_val).all(), f"F must be finite, got {F_val}"
    print(f"test_edge_i_equals_L PASSED  F={F_val.tolist()}")


if __name__ == '__main__':
    test_basic()
    test_entropy_bonus_keys_and_loss_consistency()
    test_edge_i_equals_1()
    test_edge_i_equals_L()
    test_loglik_is_q_basic()
    print("\nALL TESTS PASSED")
