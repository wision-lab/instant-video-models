#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def parse_csv_results(base_dirpath, selected_metric, nested=True):
    if nested:
        nested_dirpaths = sorted(
            list(base_dirpath.iterdir()), key=lambda p: float(p.stem)
        )
    else:
        nested_dirpaths = [base_dirpath]
    result = []
    for nested_dirpath in nested_dirpaths:
        found_metric = False
        for run_dirpath in reversed(sorted(nested_dirpath.iterdir())):
            metrics_filepath = run_dirpath / "metrics.csv"
            if not metrics_filepath.exists():
                continue
            with open(metrics_filepath) as csv_file:
                for metric_name, metric_value in csv.reader(csv_file):
                    if metric_name == selected_metric:
                        result.append(float(metric_value))
                        found_metric = True
                        break
            if found_metric:
                break
    return result


def plot_nfs_denoising_traces(plots_dirpath):
    fig, axs = plt.subplots(ncols=3, figsize=(5.1, 1.5), squeeze=True)

    experiments_dirpath = Path("outputs", "nafnet_nfs_denoising_moderate")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    axs[0].plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o",
    )
    axs[0].plot(
        parse_csv_results(output_ema_dirpath, "NormStability"),
        parse_csv_results(output_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].plot(
        parse_csv_results(ema_dirpath, "NormStability")[-1:],
        parse_csv_results(ema_dirpath, "PSNR")[-1:],
        "o-",
    )
    axs[0].plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability"),
        parse_csv_results(spatial_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].set(
        title=r"Moderate noise ($\sigma = 0.1$)", xlabel="Instability", ylabel="PSNR"
    )
    axs[0].xaxis.set_major_locator(MultipleLocator(5))

    experiments_dirpath = Path("outputs", "nafnet_nfs_denoising_strong")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    axs[1].plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o-",
    )
    axs[1].plot(
        parse_csv_results(output_ema_dirpath, "NormStability"),
        parse_csv_results(output_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[1].plot(
        parse_csv_results(ema_dirpath, "NormStability")[-1:],
        parse_csv_results(ema_dirpath, "PSNR")[-1:],
        "o-",
    )
    axs[1].plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[1].plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[1].plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability"),
        parse_csv_results(spatial_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[1].set(title=r"Strong noise ($\sigma = 0.2$)", xlabel="Instability")
    axs[1].xaxis.set_major_locator(MultipleLocator(5))

    experiments_dirpath = Path("outputs", "nafnet_nfs_denoising_extreme")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    axs[2].plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o",
        label="Base model",
    )
    axs[2].plot(
        parse_csv_results(output_ema_dirpath, "NormStability"),
        parse_csv_results(output_ema_dirpath, "PSNR"),
        "o-",
        label="Output fixed",
    )
    axs[2].plot(
        parse_csv_results(ema_dirpath, "NormStability")[-1:],
        parse_csv_results(ema_dirpath, "PSNR")[-1:],
        "o-",
        label="Simple fixed",
    )
    axs[2].plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
        label="Simple learned",
    )
    axs[2].plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
        label="Controlled",
    )
    axs[2].plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability")[:0],
        parse_csv_results(spatial_ema_dirpath, "PSNR")[:0],
        "o-",
        label="Spatial",
    )
    axs[2].set(title=r"Extreme noise ($\sigma = 0.6$)", xlabel="Instability")
    axs[2].xaxis.set_major_locator(MultipleLocator(10))
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ax in axs:
        ax.set_box_aspect(1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

    for ext in "svg", "pdf":
        fig.savefig(plots_dirpath / f"nfs_denoising_traces.{ext}")


def plot_davis_denoising_traces(plots_dirpath):
    fig, ax = plt.subplots(figsize=(2.4, 1.5), squeeze=True)

    experiments_dirpath = Path("outputs", "nafnet_davis_denoising")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    ax.plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o",
        label="Base model",
    )
    ax.plot(
        parse_csv_results(output_ema_dirpath, "NormStability"),
        parse_csv_results(output_ema_dirpath, "PSNR"),
        "o-",
        label="Output fixed",
    )
    ax.plot(
        parse_csv_results(ema_dirpath, "NormStability")[-3:],
        parse_csv_results(ema_dirpath, "PSNR")[-3:],
        "o-",
        label="Simple fixed",
    )
    ax.plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
        label="Simple learned",
    )
    ax.plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
        label="Controlled",
    )
    ax.plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability"),
        parse_csv_results(spatial_ema_dirpath, "PSNR"),
        "o-",
        label="Spatial",
    )
    ax.set(xlabel="Instability", ylabel="PSNR")
    ax.set_box_aspect(1.0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ext in "svg", "pdf":
        fig.savefig(plots_dirpath / f"davis_denoising_traces.{ext}")


def plots_laplacian_traces(plots_dirpath):
    fig, axs = plt.subplots(ncols=2, figsize=(3.6, 1.5), squeeze=True)

    experiments_dirpath = Path("outputs", "hdrnet_nfs_laplacian_moderate")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    axs[0].plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o",
    )
    axs[0].plot(
        parse_csv_results(output_ema_dirpath, "NormStability")[:4],
        parse_csv_results(output_ema_dirpath, "PSNR")[:4],
        "o-",
    )
    axs[0].plot(
        parse_csv_results(ema_dirpath, "NormStability")[:4],
        parse_csv_results(ema_dirpath, "PSNR")[:4],
        "o-",
    )
    axs[0].plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
    )
    axs[0].plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability"),
        parse_csv_results(spatial_ema_dirpath, "PSNR"),
        "o-",
    )

    axs[0].set(
        title=r"Moderate intensity ($\alpha = 0.5$)",
        xlabel="Instability",
        ylabel="PSNR",
    )

    experiments_dirpath = Path("outputs", "hdrnet_nfs_laplacian_strong")
    base_model_dirpath = experiments_dirpath / "evaluate_base_model"
    output_ema_dirpath = experiments_dirpath / "evaluate_output_simple_fixed"
    ema_dirpath = experiments_dirpath / "evaluate_simple_fixed"
    learned_ema_dirpath = experiments_dirpath / "evaluate_simple_learned"
    controlled_ema_dirpath = experiments_dirpath / "evaluate_controlled"
    spatial_ema_dirpath = experiments_dirpath / "evaluate_controlled_spatial"
    axs[1].plot(
        parse_csv_results(base_model_dirpath, "NormStability", nested=False),
        parse_csv_results(base_model_dirpath, "PSNR", nested=False),
        "o",
        label="Base model",
    )
    axs[1].plot(
        parse_csv_results(output_ema_dirpath, "NormStability")[:4],
        parse_csv_results(output_ema_dirpath, "PSNR")[:4],
        "o-",
        label="Output fixed",
    )
    axs[1].plot(
        parse_csv_results(ema_dirpath, "NormStability")[:4],
        parse_csv_results(ema_dirpath, "PSNR")[:4],
        "o-",
        label="Simple fixed",
    )
    axs[1].plot(
        parse_csv_results(learned_ema_dirpath, "NormStability"),
        parse_csv_results(learned_ema_dirpath, "PSNR"),
        "o-",
        label="Simple learned",
    )
    axs[1].plot(
        parse_csv_results(controlled_ema_dirpath, "NormStability"),
        parse_csv_results(controlled_ema_dirpath, "PSNR"),
        "o-",
        label="Controlled",
    )
    axs[1].plot(
        parse_csv_results(spatial_ema_dirpath, "NormStability"),
        parse_csv_results(spatial_ema_dirpath, "PSNR"),
        "o-",
        label="Spatial",
    )
    axs[1].set(title=r"High intensity ($\alpha = 0.25$)", xlabel="Instability")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ax in axs:
        ax.set_box_aspect(1.0)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(1))

    for ext in "svg", "pdf":
        fig.savefig(plots_dirpath / f"laplacian_traces.{ext}")


def main():
    plt.style.use(Path(__file__).parent / "paper.mplstyle")

    plots_dirpath = Path("outputs", "plots")
    plots_dirpath.mkdir(exist_ok=True)

    plot_nfs_denoising_traces(plots_dirpath)
    plot_davis_denoising_traces(plots_dirpath)
    plots_laplacian_traces(plots_dirpath)


if __name__ == "__main__":
    main()
