"""Smoke test for compute_elbo_rloo: verifies forward pass + q_theta gradient flow."""
from __future__ import annotations

import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from struct2seq.struct2seq_lo import Struct2SeqLO


def test_compute_elbo_rloo_gradient() -> None:
    torch.manual_seed(42)
    B, N = 2, 20
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
    ).to(device)

    X = torch.randn(B, N, 4, 3, device=device)
    S = torch.randint(0, 20, (B, N), device=device)
    lengths = np.array([N, N - 3])
    mask = torch.zeros(B, N, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    loss, info = model.compute_elbo_rloo(X, S, lengths, mask, num_samples=3)

    assert loss.requires_grad, "loss must require grad"
    assert loss.shape == (), "loss must be scalar"
    assert 'elbo' in info
    assert 'nll' in info
    assert 'reinforce_var' in info

    print(f"  loss          = {loss.item():.4f}")
    print(f"  ELBO          = {info['elbo'].item():.4f}")
    print(f"  NLL           = {info['nll'].item():.4f}")
    print(f"  reinforce_var = {info['reinforce_var'].item():.6f}")
    print(f"  loss_p        = {info['loss_p'].item():.4f}")
    print(f"  loss_reinforce= {info['loss_reinforce'].item():.4f}")

    loss.backward()

    q_grad_total = 0.0
    for name, p in model.named_parameters():
        if 'W_order_q' in name and p.grad is not None:
            grad_val = p.grad.abs().sum().item()
            q_grad_total += grad_val
            print(f"  grad {name}: {grad_val:.8f}")

    assert q_grad_total > 0, "W_order_q must receive non-zero gradients via REINFORCE"

    p_grad_total = 0.0
    for name, p in model.named_parameters():
        if 'W_order_p' in name and p.grad is not None:
            grad_val = p.grad.abs().sum().item()
            p_grad_total += grad_val

    assert p_grad_total > 0, "W_order_p must receive gradients"

    print("ALL CHECKS PASSED")
    print(f"  q_grad_total  = {q_grad_total:.8f}")
    print(f"  p_grad_total  = {p_grad_total:.8f}")


if __name__ == '__main__':
    test_compute_elbo_rloo_gradient()
