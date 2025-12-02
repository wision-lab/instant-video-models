#!/usr/bin/env python3

import csv
from pathlib import Path


def parse_csv_results(c, suffix=""):
    evaluation_dirpath = Path(
        "outputs",
        "nafnet_robust_spring_denoising_moderate",
        f"evaluate_spatial_ema_stabilizer_{c}{suffix}",
    )
    metrics = {}
    for run_dirpath in reversed(sorted(evaluation_dirpath.iterdir())):
        metrics_filepath = run_dirpath / "metrics.csv"
        if metrics_filepath.exists():
            with open(metrics_filepath) as f:
                for metric_name, metric_value in csv.reader(f):
                    metrics[metric_name] = float(metric_value)
            break
    return [
        f"{metrics.get('PSNR', -1.0):.2f}",
        f"{metrics.get('SSIM', -1.0):.3f}",
        f"{metrics.get('NormStability', -1.0):.2f}",
    ]


def main():
    # Parse data files.
    # The fog effect is glitchy. Frost and spatter are just static overlays.
    n = r"$\times$"
    y = r"$\checkmark$"
    stabilized_mark = [n, y, n, y]
    fine_tuned_mark = [n, n, y, y]
    variants = ["_baseline", "", "_unfrozen_ablation", "_unfrozen"]
    corruptions = ["rain", "snow"]
    lines = []
    for s, f, v in zip(stabilized_mark, fine_tuned_mark, variants):
        line = [s, f]
        for c in corruptions:
            line.extend(parse_csv_results(c, suffix=v))
        lines.append(line)

    # Generate the LaTeX table.
    table_dirpath = Path("outputs", "tables")
    table_dirpath.mkdir(parents=True, exist_ok=True)
    with open(table_dirpath / "robust_spring.tex", "w") as f:
        for line in lines:
            print(" & ".join(line) + r" \\", file=f)


if __name__ == "__main__":
    main()
