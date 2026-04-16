"""Training script for Struct2SeqLO (Learning-Order autoregressive model)."""
from __future__ import annotations, print_function

import json
import math
import os
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data.dataset import Subset

sys.path.insert(0, '..')
from struct2seq import data, noam_opt, struct2seq_lo

from utils import featurize, loss_nll, load_checkpoint, setup_device_rng, get_args


# ------------------------------------------------------------------
# Helpers for alternating (EM-style) optimisation
# ------------------------------------------------------------------

def freeze(params: Iterable[torch.nn.Parameter]) -> None:
    """Set requires_grad=False for all given parameters."""
    for p in params:
        p.requires_grad = False


def unfreeze(params: Iterable[torch.nn.Parameter]) -> None:
    """Set requires_grad=True for all given parameters."""
    for p in params:
        p.requires_grad = True


# ------------------------------------------------------------------
# Gradient diagnostics
# ------------------------------------------------------------------

def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    """L2 norm of gradients across a list of parameters."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total_sq)


def _grad_max(params: list[torch.nn.Parameter]) -> float:
    """L-inf (max absolute) of gradients across a list of parameters."""
    max_val = 0.0
    for p in params:
        if p.grad is not None:
            max_val = max(max_val, p.grad.detach().float().abs().max().item())
    return max_val


def _param_norm(params: list[torch.nn.Parameter]) -> float:
    """L2 norm of parameter values."""
    total_sq = 0.0
    for p in params:
        total_sq += p.detach().float().pow(2).sum().item()
    return math.sqrt(total_sq)


DEBUG_HEADER = (
    'Step\tPhase\t'
    'q_grad_norm\tq_grad_max\tq_param_norm\t'
    'p_grad_norm\tp_grad_max\tp_param_norm\t'
    'loss_total\tloss_rloo\tF_mean\tadv_mean\tadv_std\t'
    'q_logits_range\tlog_q_partial_mean\tlog_q_partial_std\t'
    'lr_current\n'
)


# ------------------------------------------------------------------
# Epoch helpers
# ------------------------------------------------------------------

def train_one_epoch(
    model: struct2seq_lo.Struct2SeqLO,
    loader_train: data.StructureLoader,
    optimizer: noam_opt.NoamOpt,
    device: torch.device,
    args: Any,
    start_train: float,
    total_step: int,
    beta_burial: float,
    base_folder: str,
    epoch_num: int,
    phase_label: str,
    p_params: list[torch.nn.Parameter] | None = None,
    q_params: list[torch.nn.Parameter] | None = None,
) -> tuple[int, dict[str, float]]:
    """Run one training epoch and return (updated total_step, metrics dict)."""
    model.set_burial_kl_beta(beta_burial)
    model.train()

    debug_logfile = base_folder + 'log_debug.txt'
    write_debug = (p_params is not None and q_params is not None)

    sums: dict[str, float] = {
        'elbo': 0.0, 'nll': 0.0, 'weights': 0.0,
        'entropy_norm': 0.0, 'entropy_penalty': 0.0,
        'loss_rloo': 0.0, 'loss_total': 0.0,
        'q_logits_max': 0.0, 'q_logits_min': 0.0,
        'kl_q_pi': 0.0,
    }

    for _, batch in enumerate(loader_train):
        X, S, mask, lengths = featurize(
            batch, device, shuffle_fraction=args.shuffle,
        )

        optimizer.zero_grad()
        loss, info = model.compute_elbo_paper(X, S, lengths, mask)
        loss.backward()

        # Capture gradient diagnostics AFTER backward, BEFORE step
        total_step += 1
        if write_debug :
            q_gn = _grad_norm(q_params)
            q_gm = _grad_max(q_params)
            q_pn = _param_norm(q_params)
            p_gn = _grad_norm(p_params)
            p_gm = _grad_max(p_params)
            p_pn = _param_norm(p_params)
            q_range = (
                info["q_logits_max"].item() - info["q_logits_min"].item()
            )
            with open(debug_logfile, 'a') as df:
                df.write(
                    '{}\t{}\t'
                    '{:.6e}\t{:.6e}\t{:.4f}\t'
                    '{:.6e}\t{:.6e}\t{:.4f}\t'
                    '{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t'
                    '{:.4f}\t{:.4f}\t{:.4f}\t'
                    '{:.8e}\n'.format(
                        total_step,
                        phase_label,
                        q_gn, q_gm, q_pn,
                        p_gn, p_gm, p_pn,
                        info["loss_total"].item(),
                        info["loss_rloo"].item(),
                        info.get("F_mean", torch.tensor(0.0)).item(),
                        info.get("adv_mean", torch.tensor(0.0)).item(),
                        info.get("adv_std", torch.tensor(0.0)).item(),
                        q_range,
                        info.get("log_q_partial_mean", torch.tensor(0.0)).item(),
                        info.get("log_q_partial_std", torch.tensor(0.0)).item(),
                        optimizer._rate,
                    )
                )

        optimizer.step()

        with torch.no_grad():
            log_probs = model(X, S, lengths, mask)
            _, loss_avg = loss_nll(S, log_probs, mask)

        n_tokens = mask.sum().item()
        sums['elbo'] += info["elbo"].item() * n_tokens
        sums['nll'] += loss_avg.item() * n_tokens
        sums['weights'] += n_tokens
        sums['entropy_norm'] += info["entropy_q_normalized"].item() * n_tokens
        sums['entropy_penalty'] += info["entropy_penalty"].item() * n_tokens
        sums['loss_rloo'] += info["loss_rloo"].item() * n_tokens
        sums['loss_total'] += info["loss_total"].item() * n_tokens
        sums['q_logits_max'] += info["q_logits_max"].item() * n_tokens
        sums['q_logits_min'] += info["q_logits_min"].item() * n_tokens
        sums['kl_q_pi'] += info["kl_q_pi"].item() * n_tokens

        if total_step % 100 == 0:
            elapsed = time.time() - start_train
            q_gn_str = ''
            if write_debug:
                q_gn_str = ' | q_gn {:.2e} | p_gn {:.2e}'.format(
                    _grad_norm(q_params), _grad_norm(p_params),
                )
            print(
                "[{}] Step {} | {:.0f}s | loss_total {:.4f} | loss_rloo {:.4f} | "
                "H_norm {:.4f} | ent_pen {:.4f} | ELBO {:.4f} | NLL {:.4f}"
                " | PPL {:.2f} | dF {:.4f} | q_lo [{:.4f}, {:.4f}]"
                " | KL {:.4f} | beta_b {:.4f}{}".format(
                    phase_label,
                    total_step,
                    elapsed,
                    info["loss_total"].item(),
                    info["loss_rloo"].item(),
                    info["entropy_q_normalized"].item(),
                    info["entropy_penalty"].item(),
                    info["elbo"].item(),
                    loss_avg.item(),
                    float(np.exp(loss_avg.item())),
                    info.get("delta_F_abs", torch.tensor(0.0)).item(),
                    info["q_logits_min"].item(),
                    info["q_logits_max"].item(),
                    info["kl_q_pi"].item(),
                    beta_burial,
                    q_gn_str,
                )
            )

        if total_step % 5000 == 0:
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
            }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(
                epoch_num, total_step,
            ))

    return total_step, sums


def validate_epoch(
    model: struct2seq_lo.Struct2SeqLO,
    loader_validation: data.StructureLoader,
    device: torch.device,
    args: Any,
    run_full_isq: bool,
) -> dict[str, float]:
    """Run validation and return metrics dict."""
    model.eval()
    val_proxy_sum, val_proxy_weights = 0.0, 0.0
    val_isq_sum, val_isq_weights = 0.0, 0.0

    with torch.no_grad():
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths = featurize(batch, device)

            proxy_loglik_per_res = model.compute_loglik_proxy_q_px(
                X, S, lengths, mask,
                num_samples_eval=args.proxy_num_samples,
            )
            L_proxy = torch.tensor(
                lengths, dtype=torch.float32, device=proxy_loglik_per_res.device,
            )
            val_proxy_sum += -(proxy_loglik_per_res * L_proxy).sum().cpu().item()
            val_proxy_weights += L_proxy.sum().cpu().item()

            if run_full_isq:
                isq_loglik_per_res = model.compute_loglik_is_q(
                    X, S, lengths, mask,
                    num_samples_eval=args.eval_num_samples,
                )
                L_isq = torch.tensor(
                    lengths, dtype=torch.float32, device=isq_loglik_per_res.device,
                )
                val_isq_sum += -(isq_loglik_per_res * L_isq).sum().cpu().item()
                val_isq_weights += L_isq.sum().cpu().item()

    result: dict[str, float] = {
        'proxy_sum': val_proxy_sum,
        'proxy_weights': val_proxy_weights,
    }
    if run_full_isq and val_isq_weights > 0:
        result['isq_sum'] = val_isq_sum
        result['isq_weights'] = val_isq_weights
    return result


def log_epoch(
    logfile: str,
    epoch_num: int,
    train_sums: dict[str, float],
    val_metrics: dict[str, float],
    beta_burial: float,
    run_full_isq: bool,
    phase_label: str,
) -> tuple[float, float]:
    """Compute averages, print and log one epoch. Returns (train_ppl, val_ppl_proxy)."""
    w = train_sums['weights']
    train_elbo = train_sums['elbo'] / w
    train_nll = train_sums['nll'] / w
    train_entropy_norm = train_sums['entropy_norm'] / w
    train_entropy_penalty = train_sums['entropy_penalty'] / w
    train_loss_rloo = train_sums['loss_rloo'] / w
    train_loss_total = train_sums['loss_total'] / w
    train_q_logits_max = train_sums['q_logits_max'] / w
    train_q_logits_min = train_sums['q_logits_min'] / w
    train_kl_q_pi = train_sums['kl_q_pi'] / w
    train_ppl = float(np.exp(train_nll))

    val_nll_proxy = val_metrics['proxy_sum'] / val_metrics['proxy_weights']
    val_ppl_proxy = float(np.exp(val_nll_proxy))

    if run_full_isq and 'isq_sum' in val_metrics:
        val_nll_isq = val_metrics['isq_sum'] / val_metrics['isq_weights']
        val_ppl_isq = float(np.exp(val_nll_isq))
    else:
        val_nll_isq = float('nan')
        val_ppl_isq = float('nan')

    print(
        '[{}] Epoch {} | Train ELBO {:.4f} | Train PPL {:.2f} | '
        'Val Proxy PPL {:.2f} | Val ISQ PPL {} | '
        'H_norm {:.4f} | ent_pen {:.4f} | L_rloo {:.4f} | L_total {:.4f} | '
        'q_lo [{:.4f}, {:.4f}] | KL {:.4f} | beta_b {:.4f}'.format(
            phase_label,
            epoch_num,
            train_elbo,
            train_ppl,
            val_ppl_proxy,
            '{:.2f}'.format(val_ppl_isq) if run_full_isq else 'nan',
            train_entropy_norm,
            train_entropy_penalty,
            train_loss_rloo,
            train_loss_total,
            train_q_logits_min,
            train_q_logits_max,
            train_kl_q_pi,
            beta_burial,
        )
    )

    with open(logfile, 'a') as f:
        f.write(
            '{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t'
            '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t'
            '{:.4f}\t{:.4f}\t{}\n'.format(
                epoch_num,
                train_ppl,
                val_ppl_proxy,
                val_ppl_isq,
                train_elbo,
                train_nll,
                val_nll_proxy,
                val_nll_isq,
                train_entropy_norm,
                train_entropy_penalty,
                train_loss_rloo,
                train_loss_total,
                train_q_logits_max,
                train_q_logits_min,
                train_kl_q_pi,
                beta_burial,
                phase_label,
            )
        )

    return train_ppl, val_ppl_proxy


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = get_args()

    if args.model_type != 'structure_lo':
        raise ValueError(
            "train_lo.py requires --model_type structure_lo, "
            f"got {args.model_type!r}.",
        )
    if args.eval_full_interval < 1:
        raise ValueError("--eval_full_interval must be >= 1.")

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
        num_samples=args.num_samples,
        p_encoder_arch=args.p_encoder_arch,
        p_decoder_arch=args.p_decoder_arch,
        q_encoder_arch=args.q_encoder_arch,
        q_decoder_arch=args.q_decoder_arch,
        separate_encoder=args.separate_encoder,
        separate_decoder=args.separate_decoder,
        lambda_entropy=args.lambda_entropy,
        burial_kl_beta=args.burial_kl_beta0,
        burial_kl_tau=args.burial_kl_tau,
    ).to(device)
    print('Number of parameters: {}'.format(
        sum(p.numel() for p in model.parameters())
    ))

    if args.restore != '':
        load_checkpoint(args.restore, model)

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
        f.write(
            'Epoch\tTrain_PPL\tVal_PPL_Proxy\tVal_PPL_ISQ\t'
            'Train_ELBO\tTrain_NLL\tVal_NLL_Proxy\tVal_NLL_ISQ\t'
            'Train_EntropyNorm\tTrain_EntropyPenalty\t'
            'Train_LossRLOO\tTrain_LossTotal\t'
            'Train_q_logits_max\tTrain_q_logits_min\t'
            'Train_KL_q_pi\tBeta_burial\tPhase\n'
        )
    with open(base_folder + 'args.json', 'w') as f:
        json.dump(vars(args), f)

    # Gradient debug log (always created; populated when p/q params available)
    debug_logfile = base_folder + 'log_debug.txt'
    with open(debug_logfile, 'w') as f:
        f.write(DEBUG_HEADER)

    # Pre-compute parameter lists for gradient diagnostics
    p_params = list(model.p_parameters())
    q_params = list(model.q_parameters())
    n_p = sum(p.numel() for p in p_params)
    n_q = sum(p.numel() for p in q_params)
    print(f'Parameter split: p = {n_p:,}, q = {n_q:,}')

    # ================================================================
    # Training loop
    # ================================================================
    start_train = time.time()
    epoch_losses_valid: list[float] = []
    epoch_checkpoints: list[str] = []
    total_step = 0

    if args.em_mode == 'joint':
        # ---- Joint training (original behaviour) ----
        optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)

        for e in range(args.epochs):
            epoch_num = e + 1
            if args.burial_kl_anneal_epochs > 0 and args.burial_kl_beta0 > 0:
                beta_burial = args.burial_kl_beta0 * max(
                    0.0, 1.0 - e / args.burial_kl_anneal_epochs,
                )
            else:
                beta_burial = 0.0

            total_step, train_sums = train_one_epoch(
                model, loader_train, optimizer, device, args,
                start_train, total_step, beta_burial,
                base_folder, epoch_num, phase_label='joint',
                p_params=p_params, q_params=q_params,
            )

            run_full_isq = (epoch_num == 1) or (epoch_num % args.eval_full_interval == 0)
            val_metrics = validate_epoch(
                model, loader_validation, device, args, run_full_isq,
            )

            _, val_ppl_proxy = log_epoch(
                logfile, epoch_num, train_sums, val_metrics,
                beta_burial, run_full_isq, phase_label='joint',
            )

            ckpt = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(
                epoch_num, total_step,
            )
            torch.save({
                'epoch': e,
                'hyperparams': vars(args),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
            }, ckpt)
            epoch_losses_valid.append(val_ppl_proxy)
            epoch_checkpoints.append(ckpt)

    elif args.em_mode == 'alternating':
        # ---- Variational EM (alternating p/q) ----
        optimizer_p = noam_opt.get_std_opt(p_params, args.hidden)
        optimizer_q = noam_opt.get_std_opt(
            q_params, args.hidden, factor=args.q_lr_factor,
        )

        global_epoch = 0
        q_epochs_elapsed = 0

        def _run_one_epoch(
            optimizer: noam_opt.NoamOpt,
            beta_burial: float,
            phase_label: str,
        ) -> None:
            nonlocal total_step, global_epoch
            epoch_num = global_epoch + 1

            total_step, train_sums = train_one_epoch(
                model, loader_train, optimizer, device, args,
                start_train, total_step, beta_burial,
                base_folder, epoch_num, phase_label,
                p_params=p_params, q_params=q_params,
            )

            run_full_isq = (
                (epoch_num == 1) or (epoch_num % args.eval_full_interval == 0)
            )
            val_metrics = validate_epoch(
                model, loader_validation, device, args, run_full_isq,
            )

            _, val_ppl_proxy = log_epoch(
                logfile, epoch_num, train_sums, val_metrics,
                beta_burial, run_full_isq, phase_label,
            )

            ckpt = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(
                epoch_num, total_step,
            )
            torch.save({
                'epoch': global_epoch,
                'hyperparams': vars(args),
                'model_state_dict': model.state_dict(),
                'optimizer_p_state_dict': optimizer_p.optimizer.state_dict(),
                'optimizer_q_state_dict': optimizer_q.optimizer.state_dict(),
            }, ckpt)
            epoch_losses_valid.append(val_ppl_proxy)
            epoch_checkpoints.append(ckpt)
            global_epoch += 1

        # Phase 1: p warm-up (q frozen)
        print(f'\n=== Phase 1: p warm-up for {args.warmup_p_epochs} epochs ===')
        freeze(q_params)
        unfreeze(p_params)
        for _ in range(args.warmup_p_epochs):
            if global_epoch >= args.epochs:
                break
            _run_one_epoch(optimizer_p, beta_burial=0.0, phase_label='P-warmup')

        # Phase 2: alternating EM cycles
        cycle = 0
        while global_epoch < args.epochs:
            cycle += 1

            # M-step: train p, freeze q
            print(f'\n=== EM Cycle {cycle} / M-step '
                  f'({args.epochs_per_m_step} epochs) ===')
            freeze(q_params)
            unfreeze(p_params)
            for _ in range(args.epochs_per_m_step):
                if global_epoch >= args.epochs:
                    break
                _run_one_epoch(
                    optimizer_p, beta_burial=0.0,
                    phase_label=f'M-step(c{cycle})',
                )

            # E-step: train q, freeze p
            print(f'\n=== EM Cycle {cycle} / E-step '
                  f'({args.epochs_per_e_step} epochs) ===')
            freeze(p_params)
            unfreeze(q_params)
            for _ in range(args.epochs_per_e_step):
                if global_epoch >= args.epochs:
                    break
                if args.burial_kl_anneal_epochs > 0 and args.burial_kl_beta0 > 0:
                    beta_burial = args.burial_kl_beta0 * max(
                        0.0,
                        1.0 - q_epochs_elapsed / args.burial_kl_anneal_epochs,
                    )
                else:
                    beta_burial = 0.0
                _run_one_epoch(
                    optimizer_q, beta_burial=beta_burial,
                    phase_label=f'E-step(c{cycle})',
                )
                q_epochs_elapsed += 1

        # Restore all params to trainable (clean state for evaluation)
        unfreeze(p_params)
        unfreeze(q_params)

    else:
        raise ValueError(f"Unknown em_mode: {args.em_mode!r}")

    # ================================================================
    # Best model selection & test evaluation
    # ================================================================
    best_idx = int(np.argmin(epoch_losses_valid))
    best_ckpt = epoch_checkpoints[best_idx]
    best_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_idx + 1)
    best_validation_ppl_proxy = epoch_losses_valid[best_idx]
    shutil.copy(best_ckpt, best_copy)
    load_checkpoint(best_copy, model)

    model.eval()
    with torch.no_grad():
        test_proxy_sum, test_proxy_weights = 0.0, 0.0
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)

            proxy_loglik_per_res = model.compute_loglik_proxy_q_px(
                X, S, lengths, mask,
                num_samples_eval=args.proxy_num_samples,
            )
            L_proxy = torch.tensor(
                lengths, dtype=torch.float32,
                device=proxy_loglik_per_res.device,
            )
            test_proxy_sum += -(proxy_loglik_per_res * L_proxy).sum().cpu().item()
            test_proxy_weights += L_proxy.sum().cpu().item()

    test_nll_proxy = test_proxy_sum / test_proxy_weights
    test_ppl_proxy = float(np.exp(test_nll_proxy))
    print('Perplexity\tTest Mean-NLL:{}'.format(test_ppl_proxy))

    with open(base_folder + 'results.txt', 'w') as f:
        f.write(
            'Best epoch (proxy): {}\nPerplexities:\n\tValidation: {}\n'
            '\tTest: {}'.format(
                best_idx + 1,
                best_validation_ppl_proxy,
                test_ppl_proxy,
            )
        )


if __name__ == '__main__':
    main()
