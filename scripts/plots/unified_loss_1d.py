#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.style.use(Path(__file__).parent / "paper.mplstyle")

    fig, axs = plt.subplots(
        ncols=3, figsize=(5.5, 2.0), squeeze=True, sharex=True, sharey=True
    )

    delta = 0.05
    x_samples = np.arange(-1.5, 2.5, delta)
    y_samples = np.arange(-1.0, 3.0, delta)
    x_mesh, y_mesh = np.meshgrid(x_samples, y_samples)

    pred_1 = np.array(0.0)
    true_2 = np.array(1.0)
    true_3 = np.array(2.0)

    for i, lambda_ in enumerate([0.0, 0.4, 4.0]):
        accuracy_mesh = np.abs(x_mesh - true_2) + np.abs(y_mesh - true_3)
        stability_mesh = np.abs(y_mesh - x_mesh) + np.abs(x_mesh - pred_1)
        loss_mesh = accuracy_mesh + lambda_ * stability_mesh
        axs[i].contour(x_mesh, y_mesh, loss_mesh, zorder=0, levels=16)
        axs[i].axvline(true_2, color="black", linestyle="dotted", zorder=2)
        axs[i].axhline(true_3, color="black", linestyle="dotted", zorder=2)
        axs[i].axvline(pred_1, color="black", linestyle="dashed", zorder=2)
        axs[i].axhline(pred_1, color="black", linestyle="dashed", zorder=2)
        axs[i].set_aspect("equal")
        axs[i].set_xticks([pred_1, true_2])
        axs[i].set_xticklabels([r"$\hat{y}_1$", r"$y_2$"])
        axs[i].set_yticks([pred_1, true_3])
        axs[i].set_yticklabels([r"$\hat{y}_1$", r"$y_3$"])

    axs[0].set_title(r"No stability penalty, $\lambda = 0$")
    axs[1].set_title(r"Below oracle bound, $\lambda = 0.4$")
    axs[2].set_title(r"Above collapse bound, $\lambda = 4$")
    axs[0].set_xlabel(r"Value of second prediction $\hat{y}_2$")
    axs[1].set_xlabel(r"Value of second prediction $\hat{y}_2$")
    axs[2].set_xlabel(r"Value of second prediction $\hat{y}_2$")
    axs[0].set_ylabel(r"Value of third prediction $\hat{y}_3$")

    plots_dir = Path("outputs", "plots")
    plots_dir.mkdir(exist_ok=True)
    for ext in "svg", "pdf":
        fig.savefig(plots_dir / f"unified_loss_1d.{ext}")


if __name__ == "__main__":
    main()
