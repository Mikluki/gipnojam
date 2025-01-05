import importlib.resources as pkg_resources
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import gipnojam.resources as resources
from gipnojam.core.math import (
    Noise,
    expand_down,
    expand_left,
    expand_right,
    expand_up,
    recursive_add_one,
    recursive_add_zero,
    shrink_down,
    shrink_left,
    shrink_right,
    shrink_up,
)
from gipnojam.core.polyshape import PolyShape
from gipnojam.core.polyshape_modify import rotate_polygons


def plot_matrix(matrix, show=False):
    fig, _ = plt.subplots(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, cbar=False, linewidths=0.3)
    plt.axis("equal")
    if show is False:
        plt.close(fig)


def npy_load(npy_name):
    n2array = np.load(npy_name).astype(np.int8)
    return n2array


def save_ndarray_as_npy(
    ndarray,
    export_dir,
    iter_=0,
):
    subDir = Path(export_dir, r"npy")
    subDir.mkdir(parents=True, exist_ok=True)

    fname = f"n2array_iter{iter_}.npy"
    npyPath = Path(subDir, fname)
    np.save(npyPath, ndarray)


def init_style(self):
    if self.plot_styles is None:
        plt.style.use("default")
    else:
        try:
            plt.style.use(self.plot_styles)
            # Note that styles with higher index in list will
            # overwrite values with lower index
        except Exception as e:
            print(f"# Error setting PlotterContourLog.plot_styles: {e}\n")
            plt.style.use("default")  # Fallback to default style


add_operations = [
    expand_right,
    expand_left,
    expand_up,
    expand_down,
    recursive_add_one,
]

remove_operations = [
    shrink_down,
    shrink_up,
    shrink_left,
    shrink_right,
    recursive_add_zero,
]
changes = (
    add_operations * 3
    + remove_operations * 3
    + [recursive_add_zero]
    + [recursive_add_one]
)

category = ""
EXPORT_DIR = "export" + category
UPSCALE_FACTOR = 1
LENGTH_PER_PIXEL = 1e-3

SHAPE = (64, 64)
PERLIN_PERIOD = (4, 4)

with pkg_resources.path(resources, "default_mpl_styles") as mpl_styles:
    mpl_style_color = mpl_styles / "color_palenight.mplstyle"
    mpl_style_size = mpl_styles / "size_presentation.mplstyle"
plot_styles = [str(mpl_style_color), str(mpl_style_size)]
plt.style.use(plot_styles)


# Initialize the Noise object
noise_obj = Noise(
    perlin_period=(4, 4),  # Adjust as needed
    noise_type="fractal",
    octaves=3,
    lacunarity=2,
    threshold=0.4,
)

# Generate the noise with size 64x64
noise_array = noise_obj.get_noise(shape=SHAPE, threshold=0.5)  # pyright: ignore

ps_obj = PolyShape(
    noise_array,
    lengthPerPixel=LENGTH_PER_PIXEL,
    upscale_factor=UPSCALE_FACTOR,
    fold4symmetry_poly=True,
    point8_connectivity=False,  # pyright: ignore
    cut_coners_filter=True,
    centering=False,
    save_dir=EXPORT_DIR,
    save_npy=False,
)


N_CALLS = 10
total_rotation = 0
rotation_speed = 3.5

for i in range(N_CALLS):
    change = random.choice(changes)
    print(f"{i}: {change.__name__}")

    total_rotation = rotate_polygons(
        ps_obj,
        total_rotation + rotation_speed,
        prev_rotation=total_rotation,  # pyright: ignore
    )

    ps_obj.plot_polygons(
        show=False,
        limits=True,
        color_hole="#1b1e2b",
        colormap="rainbow",
        plot_border=False,
        savefig=True,
    )

    ps_obj.apply_matrix_change(change, noise_obj=noise_obj)

# print(ps_obj.polygon_stack)
