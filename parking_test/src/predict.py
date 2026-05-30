from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import project_root


def predict(model_path: Path, source: Path, output_dir: Path, imgsz: int, conf: float, device: str | None) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model was not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Prediction source was not found: {source}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        project=str(output_dir.parent),
        name=output_dir.name,
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True,
        device=device,
    )

    print(f"Predictions saved to: {Path(results[0].save_dir) if results else output_dir}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run prediction with the fine-tuned summer YOLO model.")
    parser.add_argument("--model", type=Path, default=root / "models" / "fine_tuned_summer_best.pt")
    parser.add_argument("--source", type=Path, default=root / "dataset" / "summer_yolo" / "test" / "images")
    parser.add_argument("--output", type=Path, default=root / "runs" / "summer_predict")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        model_path=args.model.resolve(),
        source=args.source.resolve(),
        output_dir=args.output.resolve(),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
