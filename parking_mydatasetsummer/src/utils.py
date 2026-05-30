from __future__ import annotations

import shutil
from pathlib import Path

import yaml


CLASS_NAMES = ["empty", "occupied"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def write_data_yaml(path: Path, dataset_dir: Path) -> None:
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def dataset_dir_from_yaml(data_yaml: Path) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config was not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_path = data.get("path")
    if not dataset_path:
        raise RuntimeError(f"Invalid YOLO data YAML, missing 'path': {data_yaml}")

    path = Path(dataset_path)
    if not path.is_absolute():
        path = data_yaml.parent / path
    return path.resolve()


def validate_yolo_label_file(label_path: Path, class_count: int = 2) -> None:
    rows = [line.strip().split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty YOLO label file: {label_path}")

    for row in rows:
        if len(row) != 5:
            raise RuntimeError(f"Invalid YOLO row in {label_path}: {' '.join(row)}")
        class_id = int(float(row[0]))
        if class_id < 0 or class_id >= class_count:
            raise RuntimeError(f"Invalid class id {class_id} in {label_path}; expected 0..{class_count - 1}")
        coords = [float(value) for value in row[1:]]
        if not all(0.0 <= value <= 1.0 for value in coords):
            raise RuntimeError(f"Non-normalized YOLO coordinates in {label_path}")


def validate_yolo_dataset(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing YOLO folders for split '{split}': {image_dir}, {label_dir}")

        images = list_images(image_dir)
        if split in {"train", "val"} and not images:
            raise RuntimeError(f"No images found in required split: {image_dir}")

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise RuntimeError(f"Missing label for image: {image_path}")
            validate_yolo_label_file(label_path, class_count=len(CLASS_NAMES))

        counts[split] = len(images)
    return counts
