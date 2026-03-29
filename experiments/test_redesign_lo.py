"""Redesign sampling test script for Struct2SeqLO with selectable decode order.

This script mirrors `experiments/test_redesign.py`, but targets the LO-ARM model
(`--model_type structure_lo`) and allows choosing the decoding order:
  - learning_order: use the trained p_theta order head (model.sample)
  - fix_order: deterministic N->C (0..N-1) order (valid residues first)
  - any_order: random permutation over valid residues (mask==1)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass

if any(a in ("-h", "--help") for a in sys.argv):
    print(
        "\n".join(
            [
                "Usage: python experiments/test_redesign_lo.py [args]",
                "",
                "Required:",
                "  --model_type structure_lo",
                "",
                "Key args:",
                "  --restore <ckpt.pt>         Checkpoint path",
                "  --file_data <jsonl>         Dataset jsonl",
                "  --file_splits <json>        Splits json (must contain 'test')",
                "  --batch_tokens <int>        Batch size in tokens",
                "  --order_mode {learning_order,fix_order,any_order}",
                "",
                "Note: Full argument parsing is provided by experiments/utils.py.",
            ]
        )
    )
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn.functional as F

# Library code
sys.path.insert(0, "..")
from struct2seq import data, struct2seq_lo

from utils import (
    featurize,
    get_args,
    load_checkpoint,
    setup_device_rng,
    write_redesign_recovery_stat_txt,
)

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class OrderConfig:
    """Configuration for decoding order selection."""

    mode: str  # "learning_order" | "fix_order" | "any_order"


def _scores(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-protein average negative log-likelihood."""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1),
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def _S_to_seq(S: torch.Tensor, mask: torch.Tensor) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(
        alphabet[int(c)]
        for c, m in zip(S.tolist(), mask.tolist())
        if float(m) > 0
    )


def _make_full_permutation(
    mask: torch.Tensor,
    order: OrderConfig,
) -> torch.Tensor:
    """Create a [B, N] permutation with valid positions first."""
    if mask.dim() != 2:
        raise ValueError("mask must be [B, N]")
    B, N = mask.shape
    device = mask.device
    perm = torch.empty((B, N), dtype=torch.long, device=device)

    for b in range(B):
        valid_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
        invalid_idx = torch.nonzero(mask[b] <= 0.5, as_tuple=False).squeeze(-1)

        if order.mode == "fix_order":
            perm_valid = valid_idx
        elif order.mode == "any_order":
            perm_valid = valid_idx[torch.randperm(int(valid_idx.numel()), device=device)]
        else:
            raise ValueError(f"Unsupported permutation-based order_mode: {order.mode!r}")

        perm[b] = (
            torch.cat([perm_valid, invalid_idx], dim=0)
            if int(invalid_idx.numel()) > 0
            else perm_valid
        )

    return perm


def _normalize_ordering_to_full_permutation(
    ordering: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Convert a possibly partial ordering into a full [B, N] permutation.

    `Struct2SeqLO.sample()` may terminate early when no valid positions remain,
    leaving the tail of `ordering` uninitialized. For consistent scoring, we
    take the first `valid_L` decoded positions and append invalid indices.
    """
    if ordering.dim() != 2 or mask.dim() != 2:
        raise ValueError("ordering and mask must be [B, N] tensors")
    if ordering.shape != mask.shape:
        raise ValueError("ordering and mask must have the same shape")

    B, N = mask.shape
    device = mask.device
    perm_full = torch.empty((B, N), dtype=torch.long, device=device)
    valid_L = mask.sum(-1).long()

    for b in range(B):
        vL = int(valid_L[b].item())
        decoded = ordering[b, :vL]
        invalid_idx = torch.nonzero(mask[b] <= 0.5, as_tuple=False).squeeze(-1)
        if int(invalid_idx.numel()) > 0:
            perm_full[b] = torch.cat([decoded, invalid_idx], dim=0)
        else:
            perm_full[b] = decoded

    return perm_full


def _sample_with_permutation_lo(
    model: torch.nn.Module,
    X: torch.Tensor,
    lengths: np.ndarray,
    mask: torch.Tensor,
    full_perm: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Sample a sequence using a fixed full permutation via partial AR masks."""
    h_V_enc, h_E, E_idx, _ = model._encode(X, lengths, mask)  # type: ignore[attr-defined]

    B, N = mask.shape
    device = X.device
    S = torch.zeros((B, N), dtype=torch.long, device=device)

    valid_L = mask.sum(-1).long()
    max_steps = int(valid_L.max().item()) if int(B) > 0 else 0
    batch_idx_all = torch.arange(B, device=device)

    for step in range(1, max_steps + 1):
        active = valid_L >= step
        if not torch.any(active):
            break

        i_samples = torch.full((B,), step, dtype=torch.long, device=device)
        i_samples = torch.min(i_samples, valid_L)

        ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_samples)  # type: ignore[attr-defined]
        log_probs, _p_order_logits = model.forward_p(  # type: ignore[attr-defined]
            h_V_enc,
            h_E,
            E_idx,
            S,
            mask,
            ar_mask=ar_mask,
        )

        pos = full_perm[:, step - 1]
        logits_pos = log_probs[batch_idx_all, pos, :] / float(temperature)
        probs_pos = F.softmax(logits_pos, dim=-1)

        sampled = torch.multinomial(probs_pos[active], 1).squeeze(-1)
        batch_idx = batch_idx_all[active]
        pos_active = pos[active]
        S[batch_idx, pos_active] = sampled

    return S


def _similarity(seq1: str, seq2: str) -> float:
    matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
    return float(matches) / float(len(seq1))


def main() -> None:
    args = get_args()

    if args.model_type != "structure_lo":
        raise ValueError(
            "test_redesign_lo.py requires --model_type structure_lo, "
            f"got {args.model_type!r}.",
        )
    if args.restore == "":
        raise ValueError(
            "test_redesign_lo.py requires --restore <checkpoint.pt>.",
        )

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
    print(
        "Number of parameters: {}".format(
            sum(p.numel() for p in model.parameters()),
        ),
    )

    load_checkpoint(args.restore, model)

    order_mode = getattr(args, "order_mode", "learning_order")
    if order_mode not in ("learning_order", "fix_order", "any_order"):
        raise ValueError(
            "--order_mode must be 'learning_order', 'fix_order', or 'any_order' for LO."
        )
    order_cfg = OrderConfig(mode=str(order_mode))

    order_prefix = (
        "lo"
        if order_cfg.mode == "learning_order"
        else "fo"
        if order_cfg.mode == "fix_order"
        else "ao"
    )
    if args.name != "":
        base_folder = "log/" + args.name + "/" + order_prefix + "_spin2" + "/"
    else:
        base_folder = time.strftime(
            "test/%y%b%d_%I%M%p/" + order_prefix + "_spin2" + "/",
            time.localtime(),
        )

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(base_folder + "alignments", exist_ok=True)
    with open(base_folder + "/hyperparams.json", "w") as f:
        json.dump(vars(args), f)

    # Load test set
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)
    test_names: list[str] = dataset_splits["test"]
    dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
    dataset_indices = {d["name"]: i for i, d in enumerate(dataset)}
    test_set = torch.utils.data.Subset(dataset, [dataset_indices[name] for name in test_names])
    print("Testing {} domains".format(len(test_set)))

    BATCH_COPIES = 50
    NUM_BATCHES = 1
    temperatures: list[float] = [0.1] * 2

    start_time = time.time()
    total_residues = 0.0

    model.eval()
    with torch.no_grad():
        for ix, protein in enumerate(test_set):
            batch_clones = [copy.deepcopy(protein) for _ in range(BATCH_COPIES)]
            X, S_native, mask, lengths = featurize(batch_clones, device)

            log_probs_native = model(X, S_native, lengths, mask)
            native_score = float(_scores(S_native, log_probs_native, mask).cpu().numpy()[0])

            ali_file = base_folder + "alignments/" + batch_clones[0]["name"] + ".fa"
            with open(ali_file, "w") as f:
                native_seq = _S_to_seq(S_native[0], mask[0])
                f.write(f">Native, score={native_score}\n{native_seq}\n")

                for temp in temperatures:
                    for _ in range(NUM_BATCHES):
                        if order_cfg.mode == "learning_order":
                            S_sample, _ordering = model.sample(  # type: ignore[attr-defined]
                                X,
                                lengths,
                                mask,
                                temperature=float(temp),
                            )
                        else:
                            full_perm = _make_full_permutation(mask, order_cfg)
                            S_sample = _sample_with_permutation_lo(
                                model=model,
                                X=X,
                                lengths=lengths,
                                mask=mask,
                                full_perm=full_perm,
                                temperature=float(temp),
                            )

                        log_probs_sample = model(
                            X, S_sample, lengths, mask,
                        )
                        scores = _scores(S_sample, log_probs_sample, mask).cpu().numpy()

                        for b_ix in range(BATCH_COPIES):
                            seq = _S_to_seq(S_sample[b_ix], mask[0])
                            score = float(scores[b_ix])
                            f.write(f">T={float(temp)}, sample={float(b_ix)}, score={score}\n{seq}\n")

                        total_residues += float(torch.sum(mask).cpu().numpy())
                        elapsed = time.time() - start_time
                        residues_per_second = float(total_residues) / float(max(elapsed, 1e-8))
                        print(f"{residues_per_second} residues / s")

                    frac_recovery = torch.sum(mask * (S_native.eq(S_sample).float())) / torch.sum(mask)
                    print(float(frac_recovery.cpu().numpy()))

    # Aggregate results (same as test_redesign.py)
    if pd is None:
        print("pandas is not installed; skipping results aggregation and plots.")
        return

    files = [os.path.join(base_folder, "alignments", f) for f in os.listdir(base_folder + "alignments") if f.endswith(".fa")]
    rows: list[dict[str, object]] = []

    for file in files:
        with open(file, "r") as f:
            entries = f.read().split(">")[1:]
            entries = [e.strip().split("\n") for e in entries]

        native_header = entries[0][0]
        native_score = float(native_header.split(", ")[1].split("=")[1])
        native_seq = entries[0][1]

        for header, seq in entries[1:]:
            T, sample, score = [float(s.split("=")[1]) for s in header.split(", ")]
            pdb, chain = os.path.basename(file).split(".")[0:2]
            rows.append(
                {
                    "name": pdb + "." + chain,
                    "T": T,
                    "score": score,
                    "native": native_score,
                    "similarity": _similarity(native_seq, seq),
                }
            )

    df = pd.DataFrame(rows, columns=["name", "T", "score", "similarity", "native"])

    df["diff"] = -(df["score"] - df["native"])

    import matplotlib
    from matplotlib import pyplot as plt

    plt.switch_backend("agg")
    boxplot = df.boxplot(column="diff", by="T")
    plt.xlabel("Decoding temperature")
    plt.ylabel("log P(sample) - log P(native)")
    boxplot.get_figure().gca().set_title("")
    boxplot.get_figure().suptitle("")
    plt.tight_layout()
    plt.savefig(base_folder + "decoding.pdf")

    boxplot = df.boxplot(column="similarity", by="T")
    plt.xlabel("Decoding temperature")
    plt.ylabel("Native sequence recovery")
    boxplot.get_figure().gca().set_title("")
    boxplot.get_figure().suptitle("")
    plt.tight_layout()
    plt.savefig(base_folder + "recovery.pdf")

    df_mean = df.groupby(["name", "T"], as_index=False).mean()
    df_mean.to_csv(base_folder + "results.csv")
    write_redesign_recovery_stat_txt(base_folder, df)


if __name__ == "__main__":
    main()

