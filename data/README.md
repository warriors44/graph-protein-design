# Datasets

Preprocessed data and splits files are available [here](http://people.csail.mit.edu/ingraham/graph-protein-design/data/).

## Quick Download

Run the download script from this directory:

```bash
./download_data.sh
```

This downloads:
- **cath/**: `chain_set.jsonl` (~493MB), `chain_set_splits.json` - main training data
- **SPIN2/**: `test_split_L100.json`, `test_split_sc.json` - SPIN2 benchmark splits
- **ollikainen/**: `ollikainen_set.jsonl` - Ollikainen benchmark dataset
