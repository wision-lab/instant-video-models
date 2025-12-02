#!/usr/bin/env python3

import csv
from pathlib import Path


def parse_csv_results(
    dataset_variants,
    evaluation_name,
    metric_names,
    strengths,
    method_name,
    strength_format,
    metric_formats,
) -> list:
    output_dirpath = Path("outputs")
    lines = []
    for strength in strengths:
        line = [method_name, strength_format.format(strength)]
        for dataset_variant in dataset_variants:
            evaluation_dirpath = (
                output_dirpath / dataset_variant / evaluation_name / str(strength)
            )
            for run_dirpath in reversed(sorted(evaluation_dirpath.iterdir())):
                metrics_filepath = run_dirpath / "metrics.csv"
                if metrics_filepath.exists():
                    metrics_data = {}
                    with open(metrics_filepath) as metrics_file:
                        for metric_name, metric_value in csv.reader(metrics_file):
                            metrics_data[metric_name] = float(metric_value)
                    for metric_name, metric_format in zip(metric_names, metric_formats):
                        line.append(metric_format.format(metrics_data[metric_name]))
                    break
        lines.append(line)
    return lines


def write_latex_table(table_filepath, table):
    with open(table_filepath, "w") as table_file:
        for line in table:
            if isinstance(line, str):
                # noinspection PyTypeChecker
                print(line, file=table_file)
            else:
                # noinspection PyTypeChecker
                print(" & ".join(line) + r" \\", file=table_file)


def generate_denoising_table():
    dataset_variants_part_1 = [
        "nafnet_nfs_denoising_moderate",
        "nafnet_nfs_denoising_strong",
    ]
    dataset_variants_part_2 = ["nafnet_nfs_denoising_extreme", "nafnet_davis_denoising"]
    for dataset_variants, name in zip(
        (dataset_variants_part_1, dataset_variants_part_2), ("part_1", "part_2")
    ):
        table = []
        metric_names = ["NormStability", "PSNR", "SSIM"]
        alpha_strengths = [0.99, 0.98, 0.95, 0.9, 0.8, 0.6]
        alpha_strength_format = r"$\beta = {:.2f}$"
        sigma_strengths = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        sigma_strength_format = r"$\mu = {:.2f}$"
        rho_strengths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        rho_strength_format = r"$\rho = {:.2f}$"
        lambda_strengths = [0.1, 0.2, 0.4, 0.8]
        lambda_strength_format = r"$\lambda = {}$"
        metric_formats = ["{:.2f}", "{:.2f}", "{:.3f}"]
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_output_ema_stabilizer",
                metric_names,
                alpha_strengths,
                "Output EMA",
                alpha_strength_format,
                metric_formats,
            )
        )
        table.append(r"\midrule")
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_output_gaussian_stabilizer",
                metric_names,
                sigma_strengths,
                "Output Gaussian",
                sigma_strength_format,
                metric_formats,
            )
        )
        table.append(r"\midrule")
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_ema_stabilizer",
                metric_names,
                alpha_strengths,
                "Internal EMA",
                alpha_strength_format,
                metric_formats,
            )
        )
        table.append(r"\midrule")
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_learned_ema_stabilizer",
                metric_names,
                lambda_strengths,
                "Learned EMA",
                lambda_strength_format,
                metric_formats,
            )
        )
        table.append(r"\midrule")
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_controlled_ema_stabilizer",
                metric_names,
                lambda_strengths,
                "Controlled",
                lambda_strength_format,
                metric_formats,
            )
        )
        table.append(r"\midrule")
        table.extend(
            parse_csv_results(
                dataset_variants,
                "evaluate_spatial_ema_stabilizer",
                metric_names,
                lambda_strengths,
                "Spatial",
                lambda_strength_format,
                metric_formats,
            )
        )
        table_dirpath = Path("outputs", "tables")
        table_dirpath.mkdir(parents=True, exist_ok=True)
        write_latex_table(table_dirpath / f"denoising_{name}.tex", table)


def generate_depth_table():
    table = []
    dataset_variants = ["depth_anything_vision_sim", "depth_anything_sintel"]
    metric_names = ["NormStability", "DepthAbsRel", "DepthDelta-1.25"]
    alpha_strengths = [0.99, 0.98, 0.95, 0.9, 0.8, 0.6]
    alpha_strength_format = r"$\alpha = {:.2f}$"
    sigma_strengths = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    sigma_strength_format = r"$\sigma = {:.2f}$"
    rho_strengths = [0.01, 0.014, 0.018, 0.022, 0.026, 0.03]
    rho_strength_format = r"$\rho = {:.2f}$"
    lambda_strengths = [0.1, 0.2, 0.4, 0.8]
    lambda_strength_format = r"$\lambda = {}$"
    metric_formats = ["{:.2f}", "{:.3f}", "{:.3f}"]
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_output_ema_stabilizer",
            metric_names,
            alpha_strengths,
            "Output EMA",
            alpha_strength_format,
            metric_formats,
        )
    )
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_output_gaussian_stabilizer",
            metric_names,
            sigma_strengths,
            "Output Gaussian",
            sigma_strength_format,
            metric_formats,
        )
    )
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_ema_stabilizer",
            metric_names,
            alpha_strengths,
            "EMA",
            alpha_strength_format,
            metric_formats,
        )
    )
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_learned_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Learned",
            lambda_strength_format,
            metric_formats,
        )
    )
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_controlled_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Controlled",
            lambda_strength_format,
            metric_formats,
        )
    )
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_spatial_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Spatial",
            lambda_strength_format,
            metric_formats,
        )
    )
    table_dirpath = Path("outputs", "tables")
    table_dirpath.mkdir(parents=True, exist_ok=True)
    write_latex_table(table_dirpath / "depth.tex", table)


def generate_laplacian_table():
    table = []
    dataset_variants = ["hdrnet_nfs_laplacian_moderate", "hdrnet_nfs_laplacian_strong"]
    metric_names = ["NormStability", "PSNR", "SSIM"]
    alpha_strengths = [0.99, 0.98, 0.95, 0.9, 0.8, 0.6]
    alpha_strength_format = r"$\beta = {:.2f}$"
    sigma_strengths = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    sigma_strength_format = r"$\mu = {:.2f}$"
    lambda_strengths = [0.1, 0.2, 0.4, 0.8]
    lambda_strength_format = r"$\lambda = {:.1f}$"
    metric_formats = ["{:.2f}", "{:.2f}", "{:.3f}"]
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_output_ema_stabilizer",
            metric_names,
            alpha_strengths,
            "Output EMA",
            alpha_strength_format,
            metric_formats,
        )
    )
    table.append(r"\midrule")
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_output_gaussian_stabilizer",
            metric_names,
            sigma_strengths,
            "Output Gaussian",
            sigma_strength_format,
            metric_formats,
        )
    )
    table.append(r"\midrule")
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_ema_stabilizer",
            metric_names,
            alpha_strengths,
            "Internal EMA",
            alpha_strength_format,
            metric_formats,
        )
    )
    table.append(r"\midrule")
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_learned_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Learned EMA",
            lambda_strength_format,
            metric_formats,
        )
    )
    table.append(r"\midrule")
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_controlled_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Controlled",
            lambda_strength_format,
            metric_formats,
        )
    )
    table.append(r"\midrule")
    table.extend(
        parse_csv_results(
            dataset_variants,
            "evaluate_spatial_ema_stabilizer",
            metric_names,
            lambda_strengths,
            "Spatial",
            lambda_strength_format,
            metric_formats,
        )
    )
    table_dirpath = Path("outputs", "tables")
    table_dirpath.mkdir(parents=True, exist_ok=True)
    write_latex_table(table_dirpath / "laplacian.tex", table)


def main():
    generate_denoising_table()
    # generate_depth_table()
    generate_laplacian_table()


if __name__ == "__main__":
    main()
