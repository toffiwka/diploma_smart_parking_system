from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from utils import IMAGE_SUFFIXES, clear_dir, ensure_dir, project_root, validate_yolo_dataset, write_data_yaml


@dataclass(frozen=True)
class YoloPair:
    image_path: Path
    label_path: Path
    relative_key: Path


def is_metadata_file(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name == ".DS_Store" or any(part.startswith("._") for part in path.parts)


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip was not found: {zip_path}")

    clear_dir(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    roots = [path for path in extract_dir.iterdir() if path.is_dir() and not is_metadata_file(path)]
    return roots[0] if len(roots) == 1 else extract_dir


def find_custom_roots(extracted_root: Path) -> tuple[Path, Path]:
    image_roots = [path for path in extracted_root.rglob("ParkingDataset-1") if path.is_dir()]
    label_roots = [path for path in extracted_root.rglob("Labels_Output") if path.is_dir()]
    if not image_roots:
        raise RuntimeError("Could not find image folder 'ParkingDataset-1' in the extracted zip.")
    if not label_roots:
        raise RuntimeError("Could not find label folder 'Labels_Output' in the extracted zip.")
    return image_roots[0], label_roots[0]


def collect_pairs(image_root: Path, label_root: Path) -> list[YoloPair]:
    labels = {
        label_path.relative_to(label_root).with_suffix(""): label_path
        for label_path in label_root.rglob("*.txt")
        if not is_metadata_file(label_path)
    }

    pairs: list[YoloPair] = []
    for image_path in sorted(image_root.rglob("*")):
        if is_metadata_file(image_path) or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = image_path.relative_to(image_root).with_suffix("")
        label_path = labels.get(key)
        if label_path is not None:
            pairs.append(YoloPair(image_path=image_path, label_path=label_path, relative_key=key))

    if not pairs:
        raise RuntimeError(
            "No matching image/label pairs were found. Images under 'ParkingDataset-1' and labels under "
            "'Labels_Output' must have matching relative paths and filenames."
        )
    return pairs


def split_pairs(pairs: list[YoloPair], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[YoloPair]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must satisfy: 0 < train < 1, 0 <= val < 1, train + val < 1")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_dataset(splits: dict[str, list[YoloPair]], output_dir: Path) -> dict[str, int]:
    clear_dir(output_dir)
    counts: dict[str, int] = {}

    for split, pairs in splits.items():
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        ensure_dir(image_dir)
        ensure_dir(label_dir)

        for pair in pairs:
            safe_stem = "__".join(pair.relative_key.parts)
            image_name = f"{safe_stem}{pair.image_path.suffix.lower()}"
            shutil.copy2(pair.image_path, image_dir / image_name)
            shutil.copy2(pair.label_path, label_dir / f"{Path(image_name).stem}.txt")

        counts[split] = len(pairs)
    return counts


def prepare_dataset(
    zip_path: Path,
    work_dir: Path,
    output_dir: Path,
    data_yaml: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Path:
    extracted_root = extract_zip(zip_path, work_dir)
    image_root, label_root = find_custom_roots(extracted_root)
    pairs = collect_pairs(image_root, label_root)
    splits = split_pairs(pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    counts = copy_dataset(splits, output_dir)
    write_data_yaml(data_yaml, output_dir)
    validate_yolo_dataset(output_dir)

    print(f"Prepared dataset: {output_dir}")
    print(f"Data YAML: {data_yaml}")
    print(f"Images: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return data_yaml


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Prepare the custom summer YOLO zip dataset.")
    parser.add_argument("--zip", type=Path, default=Path.home() / "Downloads" / "my_dataset_summer.zip")
    parser.add_argument("--work-dir", type=Path, default=root / "dataset" / "raw_summer")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset" / "summer_yolo")
    parser.add_argument("--data-yaml", type=Path, default=root / "data_summer.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        zip_path=args.zip.resolve(),
        work_dir=args.work_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        data_yaml=args.data_yaml.resolve(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
