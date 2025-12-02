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
    x_samples = np.arange(0.0, 4.0, delta)
    y_samples = np.arange(0.0, 4.0, delta)
    x_mesh, y_mesh = np.meshgrid(x_samples, y_samples)

    pred_1 = np.array([2.0, 1.0])
    true_2 = np.array([1.0, 2.0])
    pred_3 = np.array([3.5, 3.0])

    for i, lambda_ in enumerate([0.0, 0.4, 4.0]):
        loss_mesh = np.sqrt((x_mesh - true_2[0]) ** 2 + (y_mesh - true_2[1]) ** 2)
        for neighbor in pred_1, pred_3:
            loss_mesh += lambda_ * np.sqrt(
                (x_mesh - neighbor[0]) ** 2 + (y_mesh - neighbor[1]) ** 2
            )
        axs[i].contour(x_mesh, y_mesh, loss_mesh, zorder=0, levels=16)
        axs[i].scatter(
            pred_1[0],
            pred_1[1],
            s=20,
            c="black",
            marker="<",
            zorder=2,
            label=r"$\hat{y}_1$",
        )
        axs[i].scatter(
            true_2[0],
            true_2[1],
            s=20,
            c="black",
            marker="o",
            zorder=2,
            label=r"$y_2$",
        )
        axs[i].scatter(
            pred_3[0],
            pred_3[1],
            s=20,
            c="black",
            marker=">",
            zorder=2,
            label=r"$\hat{y}_3$",
        )
        axs[i].plot(
            [pred_1[0], pred_3[0]],
            [pred_1[1], pred_3[1]],
            linestyle="dashed",
            c="black",
            zorder=1,
            label="Max stability",
        )
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    axs[0].legend()
    axs[0].set_title(r"No stability penalty, $\lambda = 0$")
    axs[1].set_title(r"Below oracle bound, $\lambda = 0.4$")
    axs[2].set_title(r"Above collapse bound, $\lambda = 4$")

    plots_dir = Path("outputs", "plots")
    plots_dir.mkdir(exist_ok=True)
    for ext in "svg", "pdf":
        fig.savefig(plots_dir / f"unified_loss_2d.{ext}")


if __name__ == "__main__":
    main()
