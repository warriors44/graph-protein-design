#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q week
#PBS -l walltime=168:00:00


cd $PBS_O_WORKDIR
source ../.venv/bin/activate

# python3 train_s2s.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full 

# python3 train_s2s.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features hbonds \
# --name h128_hbonds 

# python3 train_s2s.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features dist \
# --name h128_dist 

# python3 train_s2s.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features coarse \
# --name h128_coarse 

# python3 train_s2s.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full_mpnn \
# --mpnn 

# python3 train_lo.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full_mpnn_lo \
# --mpnn

# python3 train_lo.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full_transformer_lo 


# python3 train_ao.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full_mpnn_ao \
# --mpnn

# python3 train_ao.py \
# --cuda \
# --file_data ../data/cath/chain_set.jsonl \
# --file_splits ../data/cath/chain_set_splits.json \
# --batch_tokens 6000 \
# --features full \
# --name h128_full_transformer_ao 

python3 train_lo.py \
--cuda \
--file_data ../data/cath/chain_set.jsonl \
--file_splits ../data/cath/chain_set_splits.json \
--batch_tokens 6000 \
--features full \
--name h128_full_mpnn_lo_separate \
--mpnn \
--q_arch separate \
--epochs 300


python3 train_lo.py \
--cuda \
--file_data ../data/cath/chain_set.jsonl \
--file_splits ../data/cath/chain_set_splits.json \
--batch_tokens 6000 \
--features full \
--name h128_full_mpnn_lo_4sample \
--mpnn \
--num_samples 4 \
--epochs 300

python3 train_lo.py \
--cuda \
--file_data ../data/cath/chain_set.jsonl \
--file_splits ../data/cath/chain_set_splits.json \
--batch_tokens 6000 \
--features full \
--name h128_full_mpnn_lo_4sample_separate \
--mpnn \
--num_samples 4 \
--q_arch separate \
--epochs 300

python3 train_lo.py \
--cuda \
--file_data ../data/cath/chain_set.jsonl \
--file_splits ../data/cath/chain_set_splits.json \
--batch_tokens 6000 \
--features full \
--name h128_full_transformer_lo_4sample_separate \
--num_samples 4 \
--q_arch separate \
--epochs 300 