from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

## Library code
sys.path.insert(0, "..")
from struct2seq import data, struct2seq  # type: ignore

from matplotlib import pyplot as plt
import pandas as pd  # type: ignore

from utils import featurize, get_args, load_checkpoint, setup_device_rng


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


def similarity(seq1: str, seq2: str) -> float:
    matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
    return float(matches) / float(len(seq1))


def main() -> None:
    args = get_args()

    if args.model_type != "structure":
        raise ValueError(
            "test_redesign.py requires --model_type structure, "
            f"got {args.model_type!r}.",
        )
    if args.restore == "":
        raise ValueError("test_redesign.py requires --restore <checkpoint.pt>.")

    device = setup_device_rng(args)

    model = struct2seq.Struct2Seq(
        num_letters=args.vocab_size,
        node_features=args.hidden,
        edge_features=args.hidden,
        hidden_dim=args.hidden,
        k_neighbors=args.k_neighbors,
        protein_features=args.features,
        dropout=args.dropout,
        encoder_arch=args.p_encoder_arch,
        decoder_arch=args.p_decoder_arch,
    ).to(device)
    print(
        "Number of parameters: {}".format(
            sum(p.numel() for p in model.parameters()),
        ),
    )

    load_checkpoint(args.restore, model)

    # Load the test set from a splits file
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)
    test_names: list[str] = dataset_splits["test"]

    # Load the dataset
    dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
    dataset_indices = {d["name"]: i for i, d in enumerate(dataset)}
    test_set = torch.utils.data.Subset(
        dataset,
        [dataset_indices[name] for name in test_names],
    )
    print("Testing {} domains".format(len(test_set)))

    # Build paths for experiment
    if args.name != "":
        base_folder = "log/" + args.name + "/"
    else:
        base_folder = time.strftime("test/%y%b%d_%I%M%p/", time.localtime())

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(base_folder + "alignments", exist_ok=True)
    with open(base_folder + "/hyperparams.json", "w") as f:
        json.dump(vars(args), f)

    BATCH_COPIES = 50
    NUM_BATCHES = 1
    temperatures: list[float] = [0.1] * 2

    # Timing
    start_time = time.time()
    total_residues = 0.0

    model.eval()
    with torch.no_grad():
        for ix, protein in enumerate(test_set):
            batch_clones = [copy.deepcopy(protein) for _ in range(BATCH_COPIES)]
            X, S_native, mask, lengths = featurize(batch_clones, device)

            log_probs_native = model(X, S_native, lengths, mask)
            native_score = float(
                _scores(S_native, log_probs_native, mask).cpu().numpy()[0],
            )

            ali_file = base_folder + "alignments/" + batch_clones[0]["name"] + ".fa"
            with open(ali_file, "w") as f:
                native_seq = _S_to_seq(S_native[0], mask[0])
                f.write(f">Native, score={native_score}\n{native_seq}\n")

                for temp in temperatures:
                    for _ in range(NUM_BATCHES):
                        S_sample = model.sample(X, lengths, mask, temperature=float(temp))

                        log_probs_sample = model(X, S_sample, lengths, mask)
                        scores = _scores(S_sample, log_probs_sample, mask).cpu().numpy()

                        for b_ix in range(BATCH_COPIES):
                            seq = _S_to_seq(S_sample[b_ix], mask[0])
                            score = float(scores[b_ix])
                            f.write(
                                f">T={float(temp)}, sample={float(b_ix)}, score={score}\n{seq}\n",
                            )

                        total_residues += float(torch.sum(mask).cpu().numpy())
                        elapsed = time.time() - start_time
                        residues_per_second = float(total_residues) / float(
                            max(elapsed, 1e-8),
                        )
                        print(f"{residues_per_second} residues / s")

                    frac_recovery = torch.sum(
                        mask * (S_native.eq(S_sample).float()),
                    ) / torch.sum(mask)
                    print(float(frac_recovery.cpu().numpy()))

    # Plot the results
    files = [
        os.path.join(base_folder, "alignments", f)
        for f in os.listdir(base_folder + "alignments")
        if f.endswith(".fa")
    ]
    rows: list[dict[str, Any]] = []

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
                    "similarity": similarity(native_seq, seq),
                },
            )

    df = pd.DataFrame(rows, columns=["name", "T", "score", "similarity", "native"])
    df["diff"] = -(df["score"] - df["native"])

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

    print(f"Speed total: {residues_per_second} residues / s")
    print("Median", df_mean["similarity"].median())


if __name__ == "__main__":
    main()
