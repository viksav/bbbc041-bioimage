import argparse
from pathlib import Path

from .data import build_index
from .external_eval import evaluate_external_nih
from .external import prepare_nih_malaria
from .eval import evaluate_model
from .figures import make_figures
from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="BBBC041 malaria cell crop classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_p = subparsers.add_parser("prepare", help="Download and index BBBC041")
    prepare_p.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    prepare_p.add_argument("--seed", type=int, default=42)
    prepare_p.add_argument("--force", action="store_true")

    prepare_ext_p = subparsers.add_parser("prepare-external", help="Download NIH malaria cell images dataset")

    train_p = subparsers.add_parser("train", help="Train a transfer learning baseline")
    train_p.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=5e-4)
    train_p.add_argument("--label-smoothing", type=float, default=0.1)
    train_p.add_argument("--dropout", type=float, default=0.4)
    train_p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")
    train_p.add_argument("--freeze-backbone", action="store_true")
    train_p.add_argument("--unfreeze-layer4", action="store_true")
    train_p.add_argument("--layer4-lr-mult", type=float, default=0.01)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--max-samples", type=int, default=None)
    train_p.add_argument("--no-balance", action="store_true")
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--device", type=str, default=None)

    eval_p = subparsers.add_parser("eval", help="Evaluate and export metrics/confusion matrix")
    eval_p.add_argument("--model-path", type=Path, required=True)
    eval_p.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--max-samples", type=int, default=None)
    eval_p.add_argument("--num-workers", type=int, default=4)
    eval_p.add_argument("--device", type=str, default=None)
    eval_p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")

    fig_p = subparsers.add_parser("make-figures", help="Generate UMAP and Grad-CAM figures")
    fig_p.add_argument("--model-path", type=Path, required=True)
    fig_p.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    fig_p.add_argument("--split", choices=["train", "val", "test"], default="test")
    fig_p.add_argument("--batch-size", type=int, default=64)
    fig_p.add_argument("--embedding-samples", type=int, default=2000)
    fig_p.add_argument("--max-samples", type=int, default=None)
    fig_p.add_argument("--num-workers", type=int, default=4)
    fig_p.add_argument("--device", type=str, default=None)
    fig_p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")

    eval_ext_p = subparsers.add_parser("eval-external", help="Evaluate on NIH malaria cell images")
    eval_ext_p.add_argument("--model-path", type=Path, required=True)
    eval_ext_p.add_argument("--batch-size", type=int, default=128)
    eval_ext_p.add_argument("--num-workers", type=int, default=4)
    eval_ext_p.add_argument("--device", type=str, default=None)
    eval_ext_p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")

    args = parser.parse_args()

    if args.command == "prepare":
        build_index(label_mode=args.label_mode, seed=args.seed, force=args.force)
        return

    if args.command == "prepare-external":
        prepare_nih_malaria()
        return

    if args.command == "train":
        train_model(
            label_mode=args.label_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            dropout=args.dropout,
            backbone=args.backbone,
            freeze_backbone=args.freeze_backbone,
            unfreeze_layer4=args.unfreeze_layer4,
            layer4_lr_mult=args.layer4_lr_mult,
            seed=args.seed,
            max_samples=args.max_samples,
            balance=not args.no_balance,
            num_workers=args.num_workers,
            device=args.device,
        )
        return

    if args.command == "eval":
        evaluate_model(
            model_path=args.model_path,
            label_mode=args.label_mode,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            device=args.device,
            backbone=args.backbone,
        )
        return

    if args.command == "make-figures":
        make_figures(
            model_path=args.model_path,
            label_mode=args.label_mode,
            split=args.split,
            batch_size=args.batch_size,
            embedding_samples=args.embedding_samples,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            device=args.device,
            backbone=args.backbone,
        )
        return

    if args.command == "eval-external":
        evaluate_external_nih(
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            backbone=args.backbone,
        )
        return

if __name__ == "__main__":
    main()
