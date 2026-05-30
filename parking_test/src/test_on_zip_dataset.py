from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "runs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

from utils import CLASS_NAMES, IMAGE_SUFFIXES, clear_dir, ensure_dir, project_root


MACOS_ARTIFACTS = {"__MACOSX", ".DS_Store"}


def is_real_file(path: Path) -> bool:
    return not any(part in MACOS_ARTIFACTS or part.startswith("._") for part in path.parts)


def find_dataset_dirs(extracted_dir: Path) -> tuple[Path, Path]:
    image_dirs: list[Path] = []
    label_dirs: list[Path] = []

    for path in extracted_dir.rglob("*"):
        if not path.is_dir() or not is_real_file(path):
            continue
        lowered = path.name.lower().replace("_", " ")
        if "label" in lowered:
            label_dirs.append(path)
        if "parking" in lowered and ("dataset" in lowered or "data" in lowered):
            image_dirs.append(path)

    if not image_dirs:
        raise FileNotFoundError("Could not find an image folder with 'parking' and 'dataset' in its name.")
    if not label_dirs:
        raise FileNotFoundError("Could not find a labels folder with 'label' in its name.")

    image_dir = max(image_dirs, key=lambda p: len(list(p.rglob("*"))))
    label_dir = max(label_dirs, key=lambda p: len(list(p.rglob("*.txt"))))
    return image_dir, label_dir


def prepare_test_dataset(zip_path: Path, output_dir: Path) -> Path:
    extracted_dir = output_dir / "extracted"
    dataset_dir = output_dir / "yolo_test"
    images_out = dataset_dir / "test" / "images"
    labels_out = dataset_dir / "test" / "labels"

    clear_dir(output_dir)
    ensure_dir(extracted_dir)
    ensure_dir(images_out)
    ensure_dir(labels_out)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted_dir)

    images_root, labels_root = find_dataset_dirs(extracted_dir)
    matched = 0
    missing_labels = 0
    used_names: set[str] = set()

    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or not is_real_file(image_path):
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        relative_path = image_path.relative_to(images_root)
        label_path = labels_root / relative_path.with_suffix(".txt")
        if not label_path.exists():
            missing_labels += 1
            continue

        flat_stem = "__".join(relative_path.with_suffix("").parts)
        target_image = images_out / f"{flat_stem}{image_path.suffix.lower()}"
        target_label = labels_out / f"{flat_stem}.txt"
        if target_image.name in used_names:
            raise RuntimeError(f"Duplicate generated image name: {target_image.name}")

        used_names.add(target_image.name)
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        matched += 1

    if matched == 0:
        raise RuntimeError("No matching image/label pairs were found in the zip dataset.")

    # Ultralytics accepts train/val/test keys, and the local helper scripts expect all splits.
    for split in ["train", "val"]:
        split_images = dataset_dir / split / "images"
        split_labels = dataset_dir / split / "labels"
        ensure_dir(split_images.parent)
        shutil.copytree(images_out, split_images, dirs_exist_ok=True)
        shutil.copytree(labels_out, split_labels, dirs_exist_ok=True)

    data_yaml = output_dir / "data_new_test.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
            },
            f,
            sort_keys=False,
        )

    print(f"Images folder: {images_root}")
    print(f"Labels folder: {labels_root}")
    print(f"Matched image/label pairs: {matched}")
    print(f"Images without labels skipped: {missing_labels}")
    print(f"Prepared YOLO test dataset: {dataset_dir}")
    return data_yaml


def yolo_box_to_xyxy(row: str, width: int, height: int) -> tuple[int, np.ndarray]:
    cls, xc, yc, bw, bh = map(float, row.split())
    return int(cls), np.array(
        [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ],
        dtype=float,
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def classification_report(labels: np.ndarray, predictions: np.ndarray) -> tuple[str, np.ndarray, float]:
    matrix = confusion_matrix(labels, predictions)
    supports = matrix.sum(axis=1)
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    lines = ["              precision    recall  f1-score   support", ""]

    for idx, name in enumerate(["Empty", "Occupied"]):
        true_positive = matrix[idx, idx]
        false_positive = matrix[:, idx].sum() - true_positive
        false_negative = matrix[idx, :].sum() - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        lines.append(f"{name:>12} {precision:>10.4f} {recall:>9.4f} {f1:>9.4f} {supports[idx]:>9d}")

    accuracy = np.trace(matrix) / matrix.sum() if matrix.sum() else 0.0
    weights = supports if supports.sum() else None
    lines.extend(
        [
            "",
            f"    accuracy {'':>21} {accuracy:>9.4f} {supports.sum():>9d}",
            f"   macro avg {np.mean(precisions):>10.4f} {np.mean(recalls):>9.4f} {np.mean(f1_scores):>9.4f} {supports.sum():>9d}",
            f"weighted avg {np.average(precisions, weights=weights):>10.4f} {np.average(recalls, weights=weights):>9.4f} {np.average(f1_scores, weights=weights):>9.4f} {supports.sum():>9d}",
        ]
    )
    return "\n".join(lines), matrix, accuracy


def save_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([0, 1], ["Empty", "Occupied"])
    plt.yticks([0, 1], ["Empty", "Occupied"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_accuracy_style_test(
    model: YOLO,
    dataset_dir: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou_threshold: float,
) -> None:
    images_dir = dataset_dir / "test" / "images"
    labels_dir = dataset_dir / "test" / "labels"
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    labels: list[int] = []
    predictions: list[int] = []
    matched_labels: list[int] = []
    matched_predictions: list[int] = []
    missed = 0

    for index, image_path in enumerate(image_paths, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size

        ground_truth = [
            yolo_box_to_xyxy(line.strip(), width, height)
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        predicted_boxes: list[tuple[int, np.ndarray]] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            predicted_boxes = list(zip(classes, xyxy))

        for true_class, true_box in ground_truth:
            labels.append(true_class)
            if not predicted_boxes:
                missed += 1
                predictions.append(1 - true_class)
                continue

            ious = [box_iou(true_box, predicted_box) for _, predicted_box in predicted_boxes]
            best_index = int(np.argmax(ious))
            best_iou = float(ious[best_index])
            predicted_class = int(predicted_boxes[best_index][0])

            if best_iou >= iou_threshold:
                predictions.append(predicted_class)
                matched_labels.append(true_class)
                matched_predictions.append(predicted_class)
            else:
                missed += 1
                predictions.append(1 - true_class)

        if index % 25 == 0:
            print(f"Accuracy-style test progress: {index}/{len(image_paths)} images")

    label_array = np.array(labels, dtype=int)
    prediction_array = np.array(predictions, dtype=int)
    report, matrix, accuracy = classification_report(label_array, prediction_array)

    empty_mask = label_array == 0
    occupied_mask = label_array == 1
    lines = [
        "YOLO same-style test on ground-truth parking boxes",
        f"Images processed: {len(image_paths)}",
        f"Ground-truth boxes / patches: {len(label_array)}",
        f"Matched boxes at IoU >= {iou_threshold}: {len(matched_labels)}",
        f"Missed / low-IoU boxes counted wrong: {missed}",
        f"Accuracy only for EMPTY places: {(prediction_array[empty_mask] == 0).mean() * 100:.2f}%",
        f"Accuracy only for OCCUPIED places: {(prediction_array[occupied_mask] == 1).mean() * 100:.2f}%",
        "",
        "FINAL:",
        f"Processed patches: {len(label_array)}",
        f"Detection-aware overall accuracy: {accuracy * 100:.2f}%",
        "",
        report,
    ]

    if matched_labels:
        matched_report, _, matched_accuracy = classification_report(
            np.array(matched_labels, dtype=int),
            np.array(matched_predictions, dtype=int),
        )
        lines.extend(
            [
                "",
                "Matched-only classifier-style result:",
                f"Matched-only accuracy: {matched_accuracy * 100:.2f}%",
                matched_report,
            ]
        )

    text = "\n".join(lines)
    print("\n" + text)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.txt").write_text(text, encoding="utf-8")
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png", "YOLO Same-Style Test Confusion Matrix")


def run_tests(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    zip_path = args.zip.resolve()
    output_dir = args.output_dir.resolve()
    prepared_dir = output_dir / "prepared"

    if not model_path.exists():
        raise FileNotFoundError(f"Model was not found: {model_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip dataset was not found: {zip_path}")

    data_yaml = prepare_test_dataset(zip_path, prepared_dir)
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        split="test",
        project=str(output_dir),
        name="official_yolo_metrics",
        plots=True,
        exist_ok=True,
        device=args.device,
    )

    print("\nOfficial YOLO metrics")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall: {float(metrics.box.mr):.4f}")
    print(f"mAP@0.5: {float(metrics.box.map50):.4f}")
    print(f"mAP@0.5:0.95: {float(metrics.box.map):.4f}")
    print(f"Confusion matrix and plots: {Path(metrics.save_dir)}")

    run_accuracy_style_test(
        model=model,
        dataset_dir=prepared_dir / "yolo_test",
        output_dir=output_dir / "accuracy_style_test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou_threshold,
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test a trained YOLO model on a separate zipped YOLO dataset.")
    parser.add_argument("--zip", type=Path, required=True, help="Path to the new zipped dataset.")
    parser.add_argument("--model", type=Path, default=root / "models" / "fine_tuned_summer_best.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "new_dataset_test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None, help="Use 'cpu' or CUDA device id such as '0'.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
