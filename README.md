# GridBD-Rec

This repository provides the reference implementation of `GridBD-Rec`, including `RPQ-VAE` for `Grid-SID` generation and `SBD` for sequential recommendation in the `Grid-SID` space.

## Repository Structure

```text
GridBD-Rec/
├── main.py
├── run.sh
├── requirements.txt
├── rpq_vae/
└── sbd/
```

## Supported Datasets

The current release supports `Video_Games`, `Musical_Instruments`, and `Baby_Products`.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

To reproduce the training pipeline, prepare the data under `data/<DATASET>/` at the repository root. The current pipeline assumes precomputed item embeddings as input and trains `RPQ-VAE` and `SBD` on top of them.

For each dataset, the expected layout is:

```text
data/<DATASET>/
├── item_embeddings/
│   ├── embeddings.npy
│   └── asin2row.json
└── split/
    ├── split_train.pkl
    ├── split_val.pkl
    └── split_test.pkl
```

### Item Embeddings

`RPQ-VAE` reads item embeddings from:

```text
data/<DATASET>/item_embeddings/embeddings.npy
data/<DATASET>/item_embeddings/asin2row.json
```

`embeddings.npy` must be a NumPy array with shape `[num_items, dim]`. `asin2row.json` must map each item identifier to its row index in `embeddings.npy`. Two formats are supported:

```json
{
  "ITEM_A": 0,
  "ITEM_B": 1,
  "ITEM_C": 2
}
```

or

```json
[
  {"asin": "ITEM_A", "row": 0},
  {"asin": "ITEM_B", "row": 1},
  {"asin": "ITEM_C", "row": 2}
]
```

Row indices must be contiguous integers from `0` to `num_items - 1`.

### User Sequence Splits

`SBD` reads the user sequence splits from:

```text
data/<DATASET>/split/split_train.pkl
data/<DATASET>/split/split_val.pkl
data/<DATASET>/split/split_test.pkl
```

Each split file must be a pickled dictionary with the following structure:

```python
{
    "user_ids": ["u1", "u2", "u3"],
    "user_seqs": [
        ["item_a", "item_b", "item_c"],
        ["item_d", "item_e"],
        ["item_f", "item_g", "item_h", "item_i"],
    ],
}
```

The split files are expected to use the same item vocabulary as `asin2row.json`.

## Training

To run the full pipeline on one dataset:

```bash
bash run.sh Video_Games
bash run.sh Musical_Instruments
bash run.sh Baby_Products
```

`run.sh` forwards additional arguments to `main.py`, so stage-specific runs are also available:

```bash
bash run.sh Video_Games --stage rpq-vae
bash run.sh Musical_Instruments --stage sbd
bash run.sh Baby_Products --dry-run
```

The equivalent Python entry points are:

```bash
python main.py --dataset Video_Games
python main.py --dataset Musical_Instruments --stage rpq-vae
python main.py --dataset Baby_Products --stage sbd
```

By default, `main.py` launches `RPQ-VAE` first and then trains `SBD` on the exported `Grid-SID` files.

The SBD implementation uses PyTorch scaled dot-product attention during training and generation. Inference reuses the clean-stream KV cache, computes slot-local logits, and performs globally capped beam search with catalog compatibility maintained on the GPU. The default evaluation batch size is 64, and evaluation logs report end-to-end QPS after GPU synchronization. The attention backend can be selected with `--train_sdp_backend` and `--eval_sdp_backend` (`auto`, `math`, `flash`, or `mem_efficient`).

## Outputs

The default output structure is:

```text
outputs/<DATASET>/grid_sid/
outputs/<DATASET>/sbd/
```

`RPQ-VAE` exports the deduplicated semantic ID mapping:

```text
outputs/<DATASET>/grid_sid/asin2sid_uni.json
outputs/<DATASET>/grid_sid/meta.json
```

These files are then consumed by `SBD` during training and evaluation.
