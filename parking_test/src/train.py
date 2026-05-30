from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

from prepare_custom_dataset import prepare_dataset
from utils import dataset_dir_from_yaml, project_root, validate_yolo_dataset


def best_metrics_from_csv(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    best_row: dict[str, float] = {}
    best_map = -1.0
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {key.strip(): value.strip() for key, value in raw_row.items() if key is not None}
            try:
                current_map = float(row.get("metrics/mAP50-95(B)", "-1"))
            except ValueError:
                continue
            if current_map > best_map:
                best_map = current_map
                best_row = {
                    "precision": float(row.get("metrics/precision(B)", 0.0)),
                    "recall": float(row.get("metrics/recall(B)", 0.0)),
                    "map50": float(row.get("metrics/mAP50(B)", 0.0)),
                    "map50_95": current_map,
                }
    return best_row


def print_metrics(title: str, metrics: object) -> None:
    print(title)
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall: {float(metrics.box.mr):.4f}")
    print(f"mAP@0.5: {float(metrics.box.map50):.4f}")
    print(f"mAP@0.5:0.95: {float(metrics.box.map):.4f}")
    print("Accuracy: not directly applicable for YOLO object detection; use precision/recall/mAP.")
    print(f"Confusion matrix and plots: {Path(metrics.save_dir)}")


def train(
    data_yaml: Path,
    zip_path: Path,
    prepare: bool,
    weights: Path | str,
    imgsz: int,
    epochs: int,
    batch: int,
    patience: int,
    project: Path,
    name: str,
    device: str | None,
    workers: int,
    optimizer: str,
    lr0: float,
) -> Path:
    root = project_root()

    if prepare or not data_yaml.exists():
        prepare_dataset(
            zip_path=zip_path.resolve(),
            work_dir=root / "dataset" / "raw_summer",
            output_dir=root / "dataset" / "summer_yolo",
            data_yaml=data_yaml,
            train_ratio=0.70,
            val_ratio=0.20,
            seed=42,
        )

    validate_yolo_dataset(dataset_dir_from_yaml(data_yaml))

    model_source = str(weights)
    if isinstance(weights, Path):
        if not weights.exists():
            raise FileNotFoundError(
                f"Starting weights were not found: {weights}\n"
                "Use --weights yolov8n.pt to start from official YOLOv8n weights, or provide a local .pt checkpoint."
            )
        model_source = str(weights)

    model = YOLO(model_source)
    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=str(project),
        name=name,
        pretrained=True,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        patience=patience,
        seed=42,
        workers=workers,
        device=device,
        optimizer=optimizer,
        lr0=lr0,
    )

    run_dir = Path(results.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.exists():
        raise RuntimeError(f"Training finished but best.pt was not found at: {best_path}")

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_best = models_dir / "fine_tuned_summer_best.pt"
    final_last = models_dir / "fine_tuned_summer_last.pt"
    shutil.copy2(best_path, final_best)
    if last_path.exists():
        shutil.copy2(last_path, final_last)

    best_row = best_metrics_from_csv(run_dir / "results.csv")
    if best_row:
        print("Best validation metrics during training")
        print(f"Precision: {best_row['precision']:.4f}")
        print(f"Recall: {best_row['recall']:.4f}")
        print(f"mAP@0.5: {best_row['map50']:.4f}")
        print(f"mAP@0.5:0.95: {best_row['map50_95']:.4f}")

    best_model = YOLO(str(final_best))
    val_metrics = best_model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split="val",
        project=str(project),
        name=f"{name}_val",
        plots=True,
        exist_ok=True,
        device=device,
    )
    print_metrics("Validation metrics", val_metrics)

    test_metrics = best_model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split="test",
        project=str(project),
        name=f"{name}_test",
        plots=True,
        exist_ok=True,
        device=device,
    )
    print_metrics("Test metrics", test_metrics)

    print(f"Fine-tuned best model saved to: {final_best}")
    if final_last.exists():
        print(f"Fine-tuned last model saved to: {final_last}")
    return final_best


def parse_weights(value: str) -> Path | str:
    path = Path(value)
    return path if path.suffix == ".pt" and (path.exists() or path.parent != Path(".")) else value


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the custom summer parking dataset.")
    parser.add_argument("--data", type=Path, default=root / "data_summer.yaml")
    parser.add_argument("--zip", type=Path, default=Path.home() / "Downloads" / "my_dataset_summer.zip")
    parser.add_argument("--prepare", action="store_true", help="Rebuild dataset/summer_yolo from the zip before training.")
    parser.add_argument("--weights", type=parse_weights, default=root / "models" / "pretrained_public_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", type=Path, default=root / "runs")
    parser.add_argument("--name", type=str, default="summer_finetune")
    parser.add_argument("--device", type=str, default=None, help="Use '0' for CUDA GPU, 'cpu' for CPU, or leave empty for auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml=args.data.resolve(),
        zip_path=args.zip.resolve(),
        prepare=args.prepare,
        weights=args.weights.resolve() if isinstance(args.weights, Path) else args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        project=args.project.resolve(),
        name=args.name,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
    )
