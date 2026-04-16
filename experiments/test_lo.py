"""Evaluate Struct2SeqLO test NLL under a specified decoding order.

Supports three order modes via --order_mode:
  - fix_order:      N->C deterministic (same as FO/AO evaluation).
  - any_order:      K random permutations, arithmetic-mean NLL.
  - learning_order: K q-sampled permutations, arithmetic-mean NLL.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Subset

sys.path.insert(0, "..")
from struct2seq import data, struct2seq_lo
from struct2seq.gumbel import gumbel_top_k

from utils import featurize, loss_nll, get_args, load_checkpoint, setup_device_rng


def _make_random_permutation(mask: torch.Tensor) -> torch.Tensor:
    """Create [B, N] random permutation with valid positions first."""
    B, N = mask.shape
    device = mask.device
    perm = torch.empty((B, N), dtype=torch.long, device=device)
    for b in range(B):
        valid_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
        invalid_idx = torch.nonzero(mask[b] <= 0.5, as_tuple=False).squeeze(-1)
        perm_valid = valid_idx[torch.randperm(int(valid_idx.numel()), device=device)]
        if int(invalid_idx.numel()) > 0:
            perm[b] = torch.cat([perm_valid, invalid_idx], dim=0)
        else:
            perm[b] = perm_valid
    return perm


def _nll_with_permutation(
    model: struct2seq_lo.Struct2SeqLO,
    X: torch.Tensor,
    S: torch.Tensor,
    lengths: np.ndarray,
    mask: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[float, float]:
    """Compute total NLL and total tokens for one batch under a given permutation."""
    log_probs = model.forward(X, S, lengths, mask, permutation=permutation)
    _, loss_avg = loss_nll(S, log_probs, mask)
    n_tokens = mask.sum().item()
    return loss_avg.item() * n_tokens, n_tokens


def main() -> None:
    start_time = time.time()
    args = get_args()

    if args.model_type != "structure_lo":
        raise ValueError(
            "test_lo.py requires --model_type structure_lo, "
            f"got {args.model_type!r}.",
        )
    if args.restore == "":
        raise ValueError("test_lo.py requires --restore <checkpoint.pt>.")

    order_mode: str = getattr(args, "order_mode", "fix_order")
    if order_mode not in ("fix_order", "any_order", "learning_order"):
        raise ValueError(
            f"--order_mode must be fix_order / any_order / learning_order, got {order_mode!r}"
        )
    K: int = args.eval_num_samples

    device = setup_device_rng(args)

    model = struct2seq_lo.Struct2SeqLO(
        num_letters=args.vocab_size,
        node_features=args.hidden,
        edge_features=args.hidden,
        hidden_dim=args.hidden,
        k_neighbors=args.k_neighbors,
        protein_features=args.features,
        dropout=args.dropout,
        num_samples=args.num_samples,
        p_encoder_arch=args.p_encoder_arch,
        p_decoder_arch=args.p_decoder_arch,
        q_encoder_arch=args.q_encoder_arch,
        q_decoder_arch=args.q_decoder_arch,
        separate_encoder=args.separate_encoder,
        separate_decoder=args.separate_decoder,
    ).to(device)
    print("Number of parameters: {}".format(
        sum(p.numel() for p in model.parameters()),
    ))

    load_checkpoint(args.restore, model)

    with open(args.file_splits) as f:
        dataset_splits: dict[str, Any] = json.load(f)
    test_names = dataset_splits["test"]

    dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
    dataset_indices = {d["name"]: i for i, d in enumerate(dataset)}
    test_set = Subset(
        dataset,
        [dataset_indices[name] for name in test_names if name in dataset_indices],
    )
    loader_test = data.StructureLoader(test_set, batch_size=args.batch_tokens)
    print(f"Testing {len(test_set)} domains | order_mode={order_mode} | K={K}")

    model.eval()
    test_sum = 0.0
    test_weights = 0.0

    with torch.no_grad():
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)

            if order_mode == "fix_order":
                log_probs = model.forward(X, S, lengths, mask)
                _, loss_avg = loss_nll(S, log_probs, mask)
                n_tokens = mask.sum().item()
                test_sum += loss_avg.item() * n_tokens
                test_weights += n_tokens

            elif order_mode == "any_order":
                batch_nll = 0.0
                n_tokens = mask.sum().item()
                for _ in range(K):
                    perm = _make_random_permutation(mask)
                    nll_sum_k, _ = _nll_with_permutation(
                        model, X, S, lengths, mask, perm,
                    )
                    batch_nll += nll_sum_k
                test_sum += batch_nll / K
                test_weights += n_tokens

            elif order_mode == "learning_order":
                h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)
                if model.separate_encoder:
                    h_V_enc_q, _, _, _ = model._encode_q(X, lengths, mask)
                else:
                    h_V_enc_q = h_V_enc
                q_logits = model.forward_q(h_V_enc_q, h_E, E_idx, S, mask)

                batch_nll = 0.0
                n_tokens = mask.sum().item()
                for _ in range(K):
                    perm = gumbel_top_k(q_logits.detach())
                    nll_sum_k, _ = _nll_with_permutation(
                        model, X, S, lengths, mask, perm,
                    )
                    batch_nll += nll_sum_k
                test_sum += batch_nll / K
                test_weights += n_tokens

    test_nll = test_sum / test_weights
    test_ppl = float(np.exp(test_nll))

    print(f"[{order_mode}] Test NLL: {test_nll:.4f}  PPL: {test_ppl:.2f}")

    if args.name != '':
        base_folder = 'log/' + args.name + '/'
    else:
        base_folder = time.strftime('log/lo_%y%b%d_%I%M%p/', time.localtime())
    out_file = base_folder + f'test_{order_mode}.txt'
    with open(out_file, 'w') as f:
        f.write(f'order_mode: {order_mode}\nK: {K}\n')
        f.write(f'Test NLL: {test_nll:.4f}\nTest PPL: {test_ppl:.4f}\n')
    print(f"Results written to {out_file}")

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
