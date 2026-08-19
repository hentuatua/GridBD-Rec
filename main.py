#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


SUPPORTED_DATASETS = [
    "Video_Games",
    "Musical_Instruments",
    "Baby_Products",
]


RUNTIME_DEFAULTS = {
    "sid_gpu_ids": "0",
    "sid_nproc": 1,
    "sid_master_port": 29511,
    "block_gpu_ids": "0",
    "block_nproc": 1,
    "block_master_port": 29512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified launcher for GridBD-Rec (RPQ-VAE + SBD).")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset to launch.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "rpq-vae", "sbd"],
        help="Run RPQ-VAE only, SBD only, or both in sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"[error] {message}: {path}")


def build_rpq_vae_command(dataset: str) -> tuple[list[str], dict[str, str]]:
    out_dir = REPO_ROOT / "outputs" / dataset / "grid_sid"
    launch = ["python"]
    nproc = int(os.environ.get("SID_NPROC_PER_NODE", RUNTIME_DEFAULTS["sid_nproc"]))
    master_port = int(os.environ.get("SID_MASTER_PORT", RUNTIME_DEFAULTS["sid_master_port"]))
    if nproc > 1:
        launch = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "--master_port",
            str(master_port),
        ]

    cmd = launch + [
        str(REPO_ROOT / "rpq_vae" / "train_tokenizer.py"),
        "--dataset",
        dataset,
        "--out_dir",
        str(out_dir),
        "--device",
        "cuda",
        "--latent_dim",
        "256",
        "--num_layers",
        "3",
        "--num_splits",
        "4",
        "--codebook_size",
        "1024",
        "--layer_codebook_sizes",
        "1024,1024,1024",
        "--hidden_dims",
        "1024",
        "--dropout",
        "0.0",
        "--beta_commit",
        "0.1",
        "--load_balance_loss",
        "--load_balance_weight",
        "1e-4",
        "--load_balance_tau",
        "1.0",
        "--load_balance_eps",
        "1e-8",
        "--quant_balance_loss",
        "--quant_balance_weight",
        "1e-4",
        "--diversity_loss",
        "--diversity_weight",
        "1e-2",
        "--diversity_tau_init",
        "0.5",
        "--diversity_tau_target",
        "0.1",
        "--diversity_eps",
        "1e-8",
        "--ema_decay",
        "0.995",
        "--ema_eps",
        "1e-5",
        "--dead_restart_threshold",
        "1.0",
        "--dead_restart_noise",
        "0.03",
        "--seed",
        "42",
        "--batch_size",
        "512",
        "--lr",
        "5e-3",
        "--weight_decay",
        "1e-4",
        "--scheduler",
        "cosine",
        "--warmup_ratio",
        "0.05",
        "--min_lr_ratio",
        "0.1",
        "--max_epochs",
        "500",
        "--patience",
        "10",
        "--val_ratio",
        "0.05",
        "--grad_clip",
        "1.0",
        "--num_workers",
        "2",
        "--dedup_sid",
    ]
    cmd.append("--no-normalize")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("SID_GPU_IDS", RUNTIME_DEFAULTS["sid_gpu_ids"])
    return cmd, env


def build_sbd_command(dataset: str) -> tuple[list[str], dict[str, str]]:
    sid_dir = REPO_ROOT / "outputs" / dataset / "grid_sid"
    ckpt_name = "sbd"
    save_dir = REPO_ROOT / "outputs" / dataset / "sbd" / "checkpoints"
    log_dir = REPO_ROOT / "outputs" / dataset / "sbd" / "logs"
    log_path = log_dir / f"{ckpt_name}.log"

    launch = ["python"]
    nproc = int(os.environ.get("BLOCK_NPROC_PER_NODE", RUNTIME_DEFAULTS["block_nproc"]))
    master_port = int(os.environ.get("BLOCK_MASTER_PORT", RUNTIME_DEFAULTS["block_master_port"]))
    if nproc > 1:
        launch = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "--master_port",
            str(master_port),
        ]

    cmd = launch + [
        str(REPO_ROOT / "sbd" / "main.py"),
        "--dataset",
        dataset,
        "--train_split",
        str(REPO_ROOT / "data" / dataset / "split" / "split_train.pkl"),
        "--val_split",
        str(REPO_ROOT / "data" / dataset / "split" / "split_val.pkl"),
        "--test_split",
        str(REPO_ROOT / "data" / dataset / "split" / "split_test.pkl"),
        "--asin2sid",
        str(sid_dir / "asin2sid_uni.json"),
        "--sid_meta",
        str(sid_dir / "meta.json"),
        "--max_seq_len",
        "50",
        "--hidden_dim",
        "512",
        "--ffn_dim",
        "1024",
        "--num_heads",
        "8",
        "--num_layers",
        "4",
        "--dropout",
        "0.1",
        "--mask_prob",
        "-1",
        "--diffusion_steps",
        "1000",
        "--mask_ratio_min",
        "0.0",
        "--mask_ratio_max",
        "1.0",
        "--full_mask_ratio",
        "0.2",
        "--loss_w_seq_causal",
        "1.0",
        "--loss_w_clean_align",
        "0.25",
        "--loss_w_bidir_aux",
        "0.2",
        "--loss_w_bidir_clean_align",
        "0.25",
        "--seq_ratio_gamma",
        "1.0",
        "--seq_ratio_eps",
        "0.05",
        "--seq_ratio_clip_min",
        "0.5",
        "--seq_ratio_clip_max",
        "1.5",
        "--batch_size",
        "64",
        "--num_workers",
        "4",
        "--lr",
        "1e-3",
        "--weight_decay",
        "2e-4",
        "--num_epochs",
        "200",
        "--grad_clip",
        "1.0",
        "--scheduler",
        "cosine",
        "--warmup_ratio",
        "0.05",
        "--min_lr_ratio",
        "0.05",
        "--early_stop_patience",
        "5",
        "--eval_every",
        "5",
        "--beam_size",
        "50",
        "--decode_fill_plan",
        "1,2,1",
        "--topk_list",
        "1",
        "5",
        "10",
        "--best_metric_k",
        "10",
        "--val_eval_ratio",
        "0.1",
        "--test_eval_ratio",
        "1.0",
        "--eval_cache_batch_size",
        "64",
        "--eval_temperature",
        "1.0",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--save_dir",
        str(save_dir),
        "--log_path",
        str(log_path),
        "--ckpt_name",
        ckpt_name,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("BLOCK_GPU_IDS", RUNTIME_DEFAULTS["block_gpu_ids"])
    return cmd, env


def run_command(cmd: list[str], env: dict[str, str] | None, dry_run: bool) -> None:
    print("[launch]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "rpq-vae"):
        embed_dir = REPO_ROOT / "data" / args.dataset / "item_embeddings"
        if not args.dry_run:
            ensure_exists(embed_dir, "missing prepared item embedding directory")
            ensure_exists(embed_dir / "embeddings.npy", "missing prepared item embeddings file")
            ensure_exists(embed_dir / "asin2row.json", "missing prepared item embedding index mapping")
        cmd, env = build_rpq_vae_command(args.dataset)
        run_command(cmd, env=env, dry_run=args.dry_run)

    if args.stage in ("all", "sbd"):
        sid_dir = REPO_ROOT / "outputs" / args.dataset / "grid_sid"
        if not args.dry_run:
            ensure_exists(sid_dir / "asin2sid_uni.json", "missing Grid-SID artifact asin2sid_uni.json")
            ensure_exists(sid_dir / "meta.json", "missing Grid-SID artifact meta.json")
        cmd, env = build_sbd_command(args.dataset)
        run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
