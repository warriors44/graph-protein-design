"""Diagnostic script for compute_loglik_is_q NaN investigation.

Loads a checkpoint and runs IS-q evaluation on a few validation batches
with detailed intermediate logging to locate the source of NaN values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "..")
from struct2seq import data, struct2seq_lo
from struct2seq.gumbel import gumbel_top_k, plackett_luce_log_prob

from torch.utils.data.dataset import Subset
from utils import featurize, load_checkpoint


def _check_tensor(name: str, t: torch.Tensor) -> None:
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    finite_vals = t[torch.isfinite(t)]
    if finite_vals.numel() > 0:
        print(
            f"  {name}: shape={list(t.shape)}, nan={has_nan}, inf={has_inf}, "
            f"min={finite_vals.min().item():.6f}, max={finite_vals.max().item():.6f}, "
            f"mean={finite_vals.mean().item():.6f}"
        )
    else:
        print(
            f"  {name}: shape={list(t.shape)}, nan={has_nan}, inf={has_inf}, "
            f"ALL values are nan/inf"
        )


def diagnose_is_q(
    model: struct2seq_lo.Struct2SeqLO,
    X: torch.Tensor,
    S: torch.Tensor,
    L: np.ndarray,
    mask: torch.Tensor,
    num_samples_eval: int = 2,
) -> torch.Tensor:
    """Call model.compute_loglik_is_q and check result for NaN."""
    B, N = S.shape

    print(f"\n{'='*60}")
    print(f"Batch: B={B}, N={N}, L={L}")
    print(f"  mask.sum(-1) = {mask.sum(-1).long().tolist()}")
    print(f"{'='*60}")

    _check_tensor("X", X)
    _check_tensor("S", S.float())
    _check_tensor("mask", mask)

    loglik_per_res = model.compute_loglik_is_q(
        X, S, L, mask, num_samples_eval=num_samples_eval,
    )
    _check_tensor("loglik_per_res", loglik_per_res)

    L_tensor = torch.tensor(L, dtype=torch.float32, device=loglik_per_res.device)
    nll_batch = -(loglik_per_res * L_tensor).sum().cpu().item()
    total_tokens = L_tensor.sum().cpu().item()
    val_nll = nll_batch / total_tokens
    val_ppl = float(np.exp(val_nll))
    print(f"\n  val_nll = {val_nll:.6f}")
    print(f"  val_ppl = {val_ppl:.4f}")

    return loglik_per_res


def main() -> None:
    exp_dir = Path(
        "log/h128_full_lo_isq_e-sep_d-sep_pe-t_pd-t_qe-t_qd-t_t_n8"
    )
    args_path = exp_dir / "args.json"
    with open(args_path) as f:
        hyperparams = json.load(f)

    print("Hyperparams:", json.dumps(hyperparams, indent=2))

    ckpt_path = exp_dir / "checkpoints" / "epoch10_step6860.pt"
    if not ckpt_path.exists():
        ckpts = sorted((exp_dir / "checkpoints").glob("*.pt"))
        ckpt_path = ckpts[0]
    print(f"Loading checkpoint: {ckpt_path}")

    np.random.seed(1111)
    torch.manual_seed(1111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1111)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = struct2seq_lo.Struct2SeqLO(
        num_letters=hyperparams["vocab_size"],
        node_features=hyperparams["hidden"],
        edge_features=hyperparams["hidden"],
        hidden_dim=hyperparams["hidden"],
        k_neighbors=hyperparams["k_neighbors"],
        protein_features=hyperparams["features"],
        dropout=hyperparams["dropout"],
        num_samples=hyperparams.get("num_samples", 2),
        p_encoder_arch=hyperparams["p_encoder_arch"],
        p_decoder_arch=hyperparams["p_decoder_arch"],
        q_encoder_arch=hyperparams["q_encoder_arch"],
        q_decoder_arch=hyperparams["q_decoder_arch"],
        separate_encoder=hyperparams.get("separate_encoder", False),
        separate_decoder=hyperparams.get("separate_decoder", False),
    ).to(device)

    load_checkpoint(str(ckpt_path), model)
    model.eval()

    dataset = data.StructureDataset(
        hyperparams["file_data"], truncate=None, max_length=500,
    )
    dataset_indices = {d["name"]: i for i, d in enumerate(dataset)}
    with open(hyperparams["file_splits"]) as f:
        dataset_splits = json.load(f)

    val_set = Subset(
        dataset,
        [
            dataset_indices[name]
            for name in dataset_splits["validation"]
            if name in dataset_indices
        ],
    )
    loader_val = data.StructureLoader(val_set, batch_size=2500)

    print(f"\nValidation set: {len(val_set)} proteins")
    print(f"Running IS-q with K=2 (reduced for diagnostics)\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader_val):
            X, S, mask, lengths = featurize(batch, device)
            diagnose_is_q(model, X, S, lengths, mask, num_samples_eval=2)
            if batch_idx >= 1:
                break

    print("\n\nDiagnosis complete.")


if __name__ == "__main__":
    main()
