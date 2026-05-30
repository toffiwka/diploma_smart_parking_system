from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from utils import project_root


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP@0.5": "metrics/mAP50(B)",
    "mAP@0.5:0.95": "metrics/mAP50-95(B)",
}

LOSS_COLUMNS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def read_results(results_csv: Path) -> list[dict[str, float]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv was not found: {results_csv}")

    rows: list[dict[str, float]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                try:
                    row[key.strip()] = float(str(value).strip())
                except ValueError:
                    pass
            if row:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No metric rows could be read from: {results_csv}")
    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda row: row.get("metrics/mAP50-95(B)", -1.0))


def write_comparison_csv(output_csv: Path, first_best: dict[str, float], second_best: dict[str, float]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "first_run", "second_run", "delta"])
        for label, column in METRIC_COLUMNS.items():
            first_value = first_best.get(column, 0.0)
            second_value = second_best.get(column, 0.0)
            writer.writerow([label, f"{first_value:.6f}", f"{second_value:.6f}", f"{second_value - first_value:.6f}"])


def save_plots(
    first_rows: list[dict[str, float]],
    second_rows: list[dict[str, float]],
    first_best: dict[str, float],
    second_best: dict[str, float],
    output_dir: Path,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping comparison plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = list(METRIC_COLUMNS.keys())
    first_values = [first_best.get(METRIC_COLUMNS[name], 0.0) for name in metric_names]
    second_values = [second_best.get(METRIC_COLUMNS[name], 0.0) for name in metric_names]
    x = range(len(metric_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([idx - 0.18 for idx in x], first_values, width=0.36, label="First run")
    ax.bar([idx + 0.18 for idx in x], second_values, width=0.36, label="Second run")
    ax.set_xticks(list(x), metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("YOLOv8 metrics comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.ravel(), LOSS_COLUMNS):
        first_values = [row.get(column) for row in first_rows if column in row]
        second_values = [row.get(column) for row in second_rows if column in row]
        if first_values:
            ax.plot(range(1, len(first_values) + 1), first_values, label="First run")
        if second_values:
            ax.plot(range(1, len(second_values) + 1), second_values, label="Second run")
        ax.set_title(column)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves_comparison.png", dpi=160)
    plt.close(fig)


def compare(first_csv: Path, second_csv: Path, output_dir: Path) -> None:
    first_rows = read_results(first_csv)
    second_rows = read_results(second_csv)
    first_best = best_row(first_rows)
    second_best = best_row(second_rows)

    output_csv = output_dir / "metrics_comparison.csv"
    write_comparison_csv(output_csv, first_best, second_best)
    save_plots(first_rows, second_rows, first_best, second_best, output_dir)

    print("Best metrics comparison")
    for label, column in METRIC_COLUMNS.items():
        first_value = first_best.get(column, 0.0)
        second_value = second_best.get(column, 0.0)
        print(f"{label}: first={first_value:.4f}, second={second_value:.4f}, delta={second_value - first_value:+.4f}")
    print(f"Comparison table saved to: {output_csv}")
    print(f"Comparison plots saved under: {output_dir}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Compare two YOLO results.csv files.")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, default=root / "runs" / "summer_finetune" / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(
        first_csv=args.first.resolve(),
        second_csv=args.second.resolve(),
        output_dir=args.output_dir.resolve(),
    )
