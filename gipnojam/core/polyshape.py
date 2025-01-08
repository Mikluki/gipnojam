from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from .polyshape_geometry import matrix_to_polygons


class PolyShape:
    """
    Represents a polygon shape with various transformations and properties.

    Attributes:
        bin_matrix (np.ndarray): Input binary 2D array.
        lengthPerPixel (float): Length per pixel.
        generation_idx (int): Iterator label (default = 0).
        save_dir (str): Directory to save output (default = working directory).
        polygons (list): List of polygons and holes.
        polygonLabels (list): Labels for polygons (1 = polygon, 0 = hole).
        xc, yc (float): Center coordinates in meters.
        MATname (str): Saved matrix file name.
    """

    def __init__(
        self,
        bin_matrix_init: np.ndarray,
        lengthPerPixel: float = 1,
        upscale_factor: int = 1,
        point8_connectivity: Literal["connect", "disconnect"] = "connect",
        fold4symmetry: bool = False,
        fold4symmetry_poly: bool = False,
        cut_coners_filter: bool = False,
        centering: bool = False,
        feed_cfg: tuple = (None, None),
        generation_idx: int = 0,
        save_dir: str = "",
        # save_mat: bool = True,
        save_npy: bool = False,
    ):
        self.bin_matrix_init = bin_matrix_init
        self.lengthPerPixel = lengthPerPixel
        self.upscale_factor = upscale_factor
        self.point8_connectivity = point8_connectivity
        self.fold4symmetry = fold4symmetry
        self.fold4symmetry_poly = fold4symmetry_poly
        self.cut_coners_filter = cut_coners_filter
        self.centering = centering
        self.feed_cfg = feed_cfg
        self.generation_idx = generation_idx
        self.save_dir = save_dir
        self.save_npy = save_npy

        # Set polygon space and plot sizes
        scale_factor = 2 if fold4symmetry or fold4symmetry_poly else 1
        self.poly_space_size = (
            self.bin_matrix_init.shape[0] * scale_factor * lengthPerPixel,
            self.bin_matrix_init.shape[1] * scale_factor * lengthPerPixel,
        )
        self.poly_plot_size = (
            self.poly_space_size[0] * 1.05,
            self.poly_space_size[1] * 1.05,
        )

        # Initialize polygons and attributes
        (
            self.polygons,
            self.polygonLabels,
            self.xc,
            self.yc,
            self.bin_matrix,
        ) = self.matrix_to_polygons(self.bin_matrix_init)
        self.polygon_stack = self.polygons_to_polygon_stack(
            self.polygons, self.polygonLabels
        )
        self.coverage = self.update_coverage(self.bin_matrix)

    def add_polygon(self, poly_points, label, position="top"):
        """Add a polygon to the stack at the specified position."""
        p = np.array([[poly_points, label]], dtype=object)
        if position == "top":
            self.polygon_stack = np.vstack((self.polygon_stack, p))
        elif position == "bottom":
            self.polygon_stack = np.vstack((p, self.polygon_stack))

    def save_bin_matrix_as_npy(self):
        """Save the binary matrix as a .npy file."""
        sub_dir = Path(self.save_dir, "npy")
        sub_dir.mkdir(parents=True, exist_ok=True)
        npy_file_name = f"bin_matrix{self.bin_matrix.shape[0]}x{self.bin_matrix.shape[1]}_iter{self.generation_idx}.npy"
        npy_path = Path(sub_dir, npy_file_name)
        np.save(npy_path, self.bin_matrix)

    def matrix_to_polygons(self, bin_matrix):
        """Initialize polygons and related attributes."""
        return matrix_to_polygons(
            bin_matrix,
            lengthPerPixel=self.lengthPerPixel,
            upscale_factor=self.upscale_factor,
            point8_connectivity=self.point8_connectivity,  # pyright: ignore
            fold4symmetry=self.fold4symmetry,
            fold4symmetry_poly=self.fold4symmetry_poly,
            cut_coners_filter=self.cut_coners_filter,
            centering=self.centering,
            feed_cfg=self.feed_cfg,
        )

    def save_polygon_stack(self):
        """Save the polygon stack as a .mat file."""
        sub_dir = Path(self.save_dir, "mat")
        sub_dir.mkdir(parents=True, exist_ok=True)
        mat_file_name = f"Poly{self.bin_matrix.shape[0]}x{self.bin_matrix.shape[1]}_iter{self.generation_idx}.mat"
        polygon_stack_path = Path(sub_dir, mat_file_name)
        savemat(polygon_stack_path, {"data": self.polygon_stack})
        print(f"Polygon stack saved to: {polygon_stack_path}")
        return polygon_stack_path

    @staticmethod
    def update_coverage(matrix):
        return np.round(matrix.sum() / (matrix.shape[0] * matrix.shape[1]), 3)

    @staticmethod
    def polygons_to_polygon_stack(polygons, polygonLabels):
        """Initialize the polygon stack."""
        polygon_stack = np.empty((len(polygons), 2), dtype=object)
        for i, (polygon, label) in enumerate(zip(polygons, polygonLabels)):
            polygon_stack[i] = np.array([[polygon, label]], dtype=object)
        return polygon_stack

    def apply_matrix_change(
        self, change_function, noise_obj=None, increment_generation=True
    ):
        """
        Apply a change to the binary matrix and regenerate polygons.

        Parameters:
        -----------
        change_function : callable
            Function that modifies the binary matrix
        noise_obj : NoiseGenerator, optional
            Noise generator object to be passed to change function
        increment_generation : bool
            Whether to increment the generation index

        Returns:
        --------
        self : PolyShape
            Returns self for method chaining
        """
        # Increment generation if requested
        if increment_generation:
            self.generation_idx += 1

        # Apply change to binary matrix
        self.bin_matrix = change_function(x=self.bin_matrix, noise_obj=noise_obj)

        # Regenerate polygons from new matrix
        (
            self.polygons,
            self.polygonLabels,
            self.xc,
            self.yc,
            self.bin_matrix,
        ) = self.matrix_to_polygons(self.bin_matrix)

        # Update derived representations
        self.polygon_stack = self.polygons_to_polygon_stack(
            self.polygons, self.polygonLabels
        )
        self.coverage = self.update_coverage(self.bin_matrix)

        return self

    def change_polystack_via_func(self, change_function, *args, **kwargs):
        """
        Apply a change function to modify the polygon stack in place.

        Parameters:
        -----------
        change_function : callable
            Function that takes the current polygon stack and returns modified version
        *args, **kwargs : Additional arguments passed to the change function

        Returns:
        --------
        self : PolyShape
            Returns self for method chaining
        """
        # Apply the change function to the polygon stack
        modified_stack = []

        for poly_data in self.polygon_stack:
            # Extract polygon and label
            polygon, label = poly_data[0], poly_data[1]

            # Apply the change function to the polygon
            modified_polygon = change_function(polygon, *args, **kwargs)

            # Add modified polygon with original label
            modified_stack.append([modified_polygon, label])

        # Update the polygon stack
        self.polygon_stack = modified_stack

        # Update coverage if needed
        self.coverage = self.update_coverage(self.bin_matrix)

        return self

    def save_current_state(self):
        """Save current state to files if save_dir is set."""
        if self.save_dir:
            self.save_polygon_stack()
            if hasattr(self, "save_npy") and self.save_npy:
                self.save_bin_matrix_as_npy()
        return self

    def plot_polygons(
        self,
        limits=True,
        show=False,
        color_hole="#ffffff",
        colormap="rainbow",
        plot_border=True,
        savefig=True,
        coef=1,
        hide_ticks_n_labels=True,
    ):
        """Plot polygons with optional customization."""
        plt.rcParams.update(
            {
                "font.size": 14,
                "lines.markersize": 0,
                "lines.linewidth": 1,
                "axes.grid": False,
                "lines.marker": ".",
            }
        )

        colormaps = {
            "plasma": plt.cm.plasma,  # pyright: ignore
            "inferno": plt.cm.inferno,  # pyright: ignore
            "viridis": plt.cm.viridis,  # pyright: ignore
            "rainbow": plt.cm.rainbow,  # pyright: ignore
        }
        selected_colormap = colormaps.get(colormap, plt.cm.rainbow)  # pyright: ignore
        colors = selected_colormap(np.linspace(0, 1, len(self.polygon_stack)))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", "box")
        if limits:
            ax.set_xlim(
                -self.poly_plot_size[0] / 2 * coef,
                self.poly_plot_size[0] / 2 * coef,
            )
            ax.set_ylim(
                -self.poly_plot_size[1] / 2 * coef,
                self.poly_plot_size[1] / 2 * coef,
            )

        for polygon, color in zip(self.polygon_stack, colors):
            points, label = polygon
            ax.fill(
                points[:, 0] * coef,
                points[:, 1] * coef,
                color=color_hole if label == 0 else color,
            )

        if plot_border:
            border = plt.Rectangle(  # pyright: ignore
                (
                    self.xc * coef - self.poly_space_size[0] * coef / 2,
                    self.yc * coef - self.poly_space_size[1] * coef / 2,
                ),
                self.poly_space_size[0] * coef,
                self.poly_space_size[1] * coef,
                fc="None",
                ec="red",
                lw=1,
                alpha=0.7,
            )
            ax.add_patch(border)

        sub_dir = Path(self.save_dir, "poly")
        sub_dir.mkdir(parents=True, exist_ok=True)

        png_name = f"Poly{self.bin_matrix.shape[0]}x{self.bin_matrix.shape[1]}_epoch{self.generation_idx:04d}.png"
        self.pngPath = Path(sub_dir, png_name)

        if hide_ticks_n_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if savefig:
            fig.savefig(self.pngPath, bbox_inches="tight")
        if not show:
            plt.close(fig)
