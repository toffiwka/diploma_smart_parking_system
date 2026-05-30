from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import dataset_dir_from_yaml, project_root, validate_yolo_dataset


def evaluate(model_path: Path, data_yaml: Path, imgsz: int, split: str, project: Path, name: str, device: str | None) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model was not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_yaml}")

    validate_yolo_dataset(dataset_dir_from_yaml(data_yaml))
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split=split,
        project=str(project),
        name=name,
        plots=True,
        exist_ok=True,
        device=device,
    )

    print("Evaluation metrics")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall: {float(metrics.box.mr):.4f}")
    print(f"mAP@0.5: {float(metrics.box.map50):.4f}")
    print(f"mAP@0.5:0.95: {float(metrics.box.map):.4f}")
    print("Accuracy: not directly applicable for YOLO object detection; use precision/recall/mAP.")
    print(f"Confusion matrix and plots: {Path(metrics.save_dir)}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned summer YOLO model.")
    parser.add_argument("--model", type=Path, default=root / "models" / "fine_tuned_summer_best.pt")
    parser.add_argument("--data", type=Path, default=root / "data_summer.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--project", type=Path, default=root / "runs")
    parser.add_argument("--name", type=str, default="summer_eval")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model.resolve(),
        data_yaml=args.data.resolve(),
        imgsz=args.imgsz,
        split=args.split,
        project=args.project.resolve(),
        name=args.name,
        device=args.device,
    )
