"""Training script for Struct2SeqLO (Learning-Order autoregressive model)."""
from __future__ import print_function

import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data.dataset import Subset

sys.path.insert(0, '..')
from struct2seq import data, noam_opt, struct2seq_lo

from argparse import ArgumentParser
from utils import featurize, loss_nll, load_checkpoint, setup_device_rng


def get_args_lo() -> ArgumentParser:
    parser = ArgumentParser(description='Learning-Order Struct2Seq training')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--k_neighbors', type=int, default=30)
    parser.add_argument('--vocab_size', type=int, default=20)
    parser.add_argument('--features', type=str, default='full')
    parser.add_argument('--mpnn', action='store_true')

    parser.add_argument('--restore', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--file_data', type=str, default='../data/cath/chain_set.jsonl')
    parser.add_argument('--file_splits', type=str, default='../data/cath/chain_set_splits.json')
    parser.add_argument('--batch_tokens', type=int, default=2500)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--shuffle', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of permutation samples K for RLOO')
    return parser.parse_args()


def main() -> None:
    args = get_args_lo()
    device = setup_device_rng(args)

    # Build model
    model = struct2seq_lo.Struct2SeqLO(
        num_letters=args.vocab_size,
        node_features=args.hidden,
        edge_features=args.hidden,
        hidden_dim=args.hidden,
        k_neighbors=args.k_neighbors,
        protein_features=args.features,
        dropout=args.dropout,
        use_mpnn=args.mpnn,
    ).to(device)
    print('Number of parameters: {}'.format(
        sum(p.numel() for p in model.parameters())
    ))

    if args.restore != '':
        load_checkpoint(args.restore, model)

    optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)

    # Dataset
    dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
    dataset_indices = {d['name']: i for i, d in enumerate(dataset)}
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)

    train_set, validation_set, test_set = [
        Subset(dataset, [
            dataset_indices[name]
            for name in dataset_splits[key]
            if name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]
    loader_train, loader_validation, loader_test = [
        data.StructureLoader(d, batch_size=args.batch_tokens)
        for d in [train_set, validation_set, test_set]
    ]
    print('Training:{}, Validation:{}, Test:{}'.format(
        len(train_set), len(validation_set), len(test_set),
    ))

    # Logging
    if args.name != '':
        base_folder = 'log/' + args.name + '/'
    else:
        base_folder = time.strftime('log/lo_%y%b%d_%I%M%p/', time.localtime())
    os.makedirs(base_folder, exist_ok=True)
    for sub in ['checkpoints']:
        os.makedirs(base_folder + sub, exist_ok=True)

    logfile = base_folder + 'log.txt'
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain_ELBO\tTrain_NLL\tVal_NLL\n')
    with open(base_folder + 'args.json', 'w') as f:
        json.dump(vars(args), f)

    # Training loop
    start_train = time.time()
    epoch_losses_valid: list[float] = []
    epoch_checkpoints: list[str] = []
    total_step = 0

    for e in range(args.epochs):
        model.train()
        train_elbo_sum, train_nll_sum, train_weights = 0., 0., 0.

        for train_i, batch in enumerate(loader_train):
            X, S, mask, lengths = featurize(
                batch, device, shuffle_fraction=args.shuffle,
            )

            optimizer.zero_grad()
            loss, info = model.compute_elbo_rloo(
                X, S, lengths, mask, num_samples=args.num_samples,
            )

            loss.backward()
            optimizer.step()

            total_step += 1
            n_tokens = mask.sum().item()
            train_elbo_sum += info['elbo'].item() * n_tokens
            train_nll_sum += info['nll'].item() * n_tokens
            train_weights += n_tokens

            if total_step % 100 == 0:
                elapsed = time.time() - start_train
                print(
                    'Step {} | {:.0f}s | ELBO {:.4f} | NLL {:.4f}'
                    ' | PPL {:.2f} | RF_var {:.4f}'.format(
                        total_step, elapsed,
                        info['elbo'].item(), info['nll'].item(),
                        np.exp(info['nll'].item()),
                        info.get('reinforce_var', torch.tensor(0.0)).item(),
                    )
                )

            if total_step % 5000 == 0:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e + 1, total_step))

        # Validation
        model.eval()
        with torch.no_grad():
            val_sum, val_weights = 0., 0.
            for _, batch in enumerate(loader_validation):
                X, S, mask, lengths = featurize(batch, device)
                log_probs = model(X, S, lengths, mask)
                loss_per_res, loss_avg = loss_nll(S, log_probs, mask)
                val_sum += torch.sum(loss_per_res * mask).cpu().item()
                val_weights += torch.sum(mask).cpu().item()

        train_elbo = train_elbo_sum / train_weights
        train_nll = train_nll_sum / train_weights
        val_nll = val_sum / val_weights
        val_ppl = np.exp(val_nll)
        train_ppl = np.exp(train_nll)

        print('Epoch {} | Train ELBO {:.4f} | Train PPL {:.2f} | Val PPL {:.2f}'.format(
            e + 1, train_elbo, train_ppl, val_ppl,
        ))

        with open(logfile, 'a') as f:
            f.write('{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                e + 1, train_elbo, train_nll, val_nll,
            ))

        ckpt = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e + 1, total_step)
        torch.save({
            'epoch': e,
            'hyperparams': vars(args),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
        }, ckpt)

        epoch_losses_valid.append(val_ppl)
        epoch_checkpoints.append(ckpt)

    # Best model
    best_idx = int(np.argmin(epoch_losses_valid))
    best_ckpt = epoch_checkpoints[best_idx]
    best_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_idx + 1)
    shutil.copy(best_ckpt, best_copy)
    load_checkpoint(best_copy, model)

    # Test
    model.eval()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)
            log_probs = model(X, S, lengths, mask)
            loss_per_res, _ = loss_nll(S, log_probs, mask)
            test_sum += torch.sum(loss_per_res * mask).cpu().item()
            test_weights += torch.sum(mask).cpu().item()

    test_ppl = np.exp(test_sum / test_weights)
    print('Test Perplexity: {:.2f}'.format(test_ppl))

    with open(base_folder + 'results.txt', 'w') as f:
        f.write('Best epoch: {}\nTest perplexity: {:.4f}\n'.format(
            best_idx + 1, test_ppl,
        ))


if __name__ == '__main__':
    main()
