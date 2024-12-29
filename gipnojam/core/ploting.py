import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def add_text(ax, x, y, text_str, fontsize=None, horizontalalignment="left"):
    ax.text(
        x,
        y,
        text_str,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment=horizontalalignment,
        transform=ax.transAxes,
    )


def plot_matrix(
    matrix, png_name, png_save_dir, figsize=(16, 16), font_scale=0.9
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=font_scale)
    sns.heatmap(
        matrix, annot=True, cbar=False, linewidths=0.3, cmap="rocket_r"
    )
    plt.axis("equal")
    plt.title(png_name)
    png_save_path = Path(png_save_dir, png_name)
    fig.savefig(png_save_path, bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_absorption_coef(
    ant_obj,
    leftLim,
    rightLim,
    coef=1e-9,
    font_size=20,
    figsize=(12, 10),
):
    plt.rcParams.update(
        {
            "font.size": font_size,
            "lines.linewidth": 2,
            "lines.marker": ".",
            "lines.markersize": 0,
            "axes.grid": True,
            "axes.autolimit_mode": "round_numbers",
        }
    )
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    dashline_color = "#292d3e"
    fill_color = "#9f9f9f"
    fill_alpha = 0.2
    for idx in ant_obj.idxCross:
        ax2.axvline(
            ant_obj.frequencyRange[idx] * coef,
            lw=2,
            linestyle="--",
            color=dashline_color,
        )
    ax2.axvspan(
        leftLim * coef,
        rightLim * coef,
        color=fill_color,
        alpha=fill_alpha,
    )

    ax2.plot(
        ant_obj.frequencyRange * coef,
        ant_obj.absorption_coef,
        lw=3,
        label=None,
    )

    if ant_obj.lossVal != None:
        fig.suptitle(
            f"iter = {ant_obj.iter_}, Loss = {ant_obj.lossVal:1.0f}"
        )

    ax2.set_title(
        f"Absorption, MatrixSize = {ant_obj.shape_obj.bin_matrix.shape} x 2*{ant_obj.shape_obj.lengthPerPixel}",
    )  # fontsize=24
    ax2.set_xlabel("Frequency (GHz)")
    text_str = f"frequency_points_num = {len(ant_obj.frequencyRange)}"
    add_text(ax2, 0.01, 0.01, text_str)
    # ax2.set_ylabel('')
    ax2.set_ylim(0, 1)

    if ant_obj.lossVal != None:
        pngName = f"absorption_iter{ant_obj.iter_}_Loss_{ant_obj.lossVal:1.0f}.png"
    else:
        pngName = f"absorption_iter{ant_obj.iter_}.png"

    ant_obj.absorption_save_dir = Path(ant_obj.SAVE_DIR, r"absorption")
    ant_obj.absorption_save_dir.mkdir(parents=True, exist_ok=True)

    ant_obj.absorption_pngPath = Path(ant_obj.absorption_save_dir, pngName)
    fig.savefig(ant_obj.absorption_pngPath, bbox_inches="tight")
    plt.close(fig)


def plot_z(
    ant_obj,
    leftLim,
    rightLim,
    coef=1e-9,
    font_size=20,
    figsize=(12, 10),
):
    plt.rcParams.update(
        {
            "font.size": font_size,
            "lines.linewidth": 2,
            "lines.marker": ".",
            "lines.markersize": 0,
            "axes.grid": True,
            "axes.autolimit_mode": "round_numbers",
        }
    )
    # 'axes.prop_cycle': plt.cycler(color=colors), 'figure.max_open_warning': 0})
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    dashline_color = "#292d3e"
    fill_color = "#9f9f9f"
    fill_alpha = 0.2
    for idx in ant_obj.idxCross:
        ax1.axvline(
            ant_obj.frequencyRange[idx] * coef,
            lw=2,
            linestyle="--",
            color=dashline_color,
        )
    ax1.axvspan(
        leftLim * coef,
        rightLim * coef,
        color=fill_color,
        alpha=fill_alpha,
    )

    ax1.plot(
        ant_obj.frequencyRange * coef,
        ant_obj.real_Z,
        lw=3,
        label="Re(Z)",
    )
    ax1.plot(
        ant_obj.frequencyRange * coef,
        ant_obj.imaginary_Z,
        lw=3,
        label="Im(Z)",
    )

    ax1.set_title(
        "Z",
    )  # weight='bold', fontsize=24
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Impedance (Ohms)")
    ax1.axhline(0, lw=1.5, color="black")
    ax1.legend()

    if ant_obj.lossVal != None:
        pngName = f"z_iter{ant_obj.iter_}_Loss_{ant_obj.lossVal:1.0f}.png"
    else:
        pngName = f"z_iter{ant_obj.iter_}.png"

    ant_obj.q = Path(ant_obj.SAVE_DIR, r"z")
    ant_obj.z_save_dir.mkdir(parents=True, exist_ok=True)

    ant_obj.z_pngPath = Path(ant_obj.z_save_dir, pngName)
    fig.savefig(ant_obj.z_pngPath, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(opt_obj, figsize=(12, 10)):
    plt.rcParams.update(
        {
            "font.size": 10,
            "lines.markersize": 10,
            "lines.linewidth": 2,
            "axes.grid": True,
            "lines.marker": ".",
        }
    )
    iterations = range(1, opt_obj.n_calls + 1)
    mins = [np.min(opt_obj.ySteps[:i]) for i in iterations]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, mins)
    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x_ini)$ after $n$ calls")
    pngName = "Convergence_plot.png"
    fig.savefig(Path(opt_obj.EXPORT_DIR, pngName), bbox_inches="tight")
