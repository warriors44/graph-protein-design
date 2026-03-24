"""Evaluate Struct2SeqLO on the test split using IS-q (q-based importance sampling)."""
from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Subset

sys.path.insert(0, "..")
from struct2seq import data, struct2seq_lo

from utils import featurize, get_args, load_checkpoint, setup_device_rng


def main() -> None:
    args = get_args()

    if args.model_type != "structure_lo":
        raise ValueError(
            "test_lo.py requires --model_type structure_lo, "
            f"got {args.model_type!r}.",
        )
    if args.restore == "":
        raise ValueError("test_lo.py requires --restore <checkpoint.pt>.")

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
    print("Testing {} domains".format(len(test_set)))

    model.eval()
    test_isq_sum = 0.0
    test_isq_weights = 0.0
    with torch.no_grad():
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)

            is_q_loglik_per_res = model.compute_loglik_is_q(
                X,
                S,
                lengths,
                mask,
                num_samples_eval=args.eval_num_samples,
            )
            L_tensor_isq = torch.tensor(
                lengths,
                dtype=torch.float32,
                device=is_q_loglik_per_res.device,
            )
            is_q_nll_batch = -(is_q_loglik_per_res * L_tensor_isq).sum().cpu().item()
            test_isq_sum += is_q_nll_batch
            test_isq_weights += L_tensor_isq.sum().cpu().item()

    test_nll_isq = test_isq_sum / test_isq_weights
    test_ppl_isq = float(np.exp(test_nll_isq))

    if args.name != '':
        base_folder = 'log/' + args.name + '/'
    else:
        base_folder = time.strftime('log/lo_%y%b%d_%I%M%p/', time.localtime())
    with open(base_folder + 'test.txt', 'w') as f:
        f.write('Perplexity\tTest ISQ:{}\n'.format(test_ppl_isq))

if __name__ == "__main__":
    main()
