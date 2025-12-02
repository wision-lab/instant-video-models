#!/usr/bin/env python3

import csv
from pathlib import Path


def parse_csv_results(c1, c2, suffix=""):
    evaluation_dirpath = Path(
        "outputs", "evaluate_stabilizer_composition", f"{c1}_{c2}{suffix}"
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
    corruptions = [
        "chunk_drop",
        "elastic_transform",
        "frame_drop",
        "jpeg_compression",
        "salt_pepper_noise",
    ]
    names = [
        "Patch drop",
        "Elastic distortion",
        "Frame drop",
        "JPEG artifacts",
        "Impulse noise",
    ]
    lines = []
    for c1, n1 in zip(corruptions, names):
        for c2, n2 in zip(corruptions, names):
            if c1 == c2:
                continue
            line = [n1, n2]
            line.extend(parse_csv_results(c1, c2))
            line.extend(parse_csv_results(c1, c2, suffix="_baseline"))
            lines.append(line)

    # Generate the LaTeX table.
    table_dirpath = Path("outputs", "tables")
    table_dirpath.mkdir(parents=True, exist_ok=True)
    with open(table_dirpath / "stabilizer_composition.tex", "w") as f:
        for line in lines:
            s = line if isinstance(line, str) else (" & ".join(line) + r" \\")
            print(s, file=f)


if __name__ == "__main__":
    main()
