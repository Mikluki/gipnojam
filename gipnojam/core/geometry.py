from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from skimage import measure


def count_coverage(matrix):
    return np.round(matrix.sum() / (matrix.shape[0] * matrix.shape[1]), 3)


def rectangle(x, y, xc, yc):
    rec = np.array(
        [
            [x / 2, y / 2],
            [x / 2, -y / 2],
            [-x / 2, -y / 2],
            [-x / 2, y / 2],
        ]
    )
    rec[:, 0] += xc
    rec[:, 1] += yc
    return rec


def circle(r, xc=0, yc=0, num_points=40):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = r * np.cos(theta) + xc
    y = r * np.sin(theta) + yc
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))


def get_all_edges(bin_matrix):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bin_matrix)
    for i, j in zip(ii, jj):
        # North
        if j == bin_matrix.shape[1] - 1 or not bin_matrix[i, j + 1]:
            edges.append(np.array([[i, j + 1], [i + 1, j + 1]]))
        # East
        if i == bin_matrix.shape[0] - 1 or not bin_matrix[i + 1, j]:
            edges.append(np.array([[i + 1, j], [i + 1, j + 1]]))
        # South
        if j == 0 or not bin_matrix[i, j - 1]:
            edges.append(np.array([[i, j], [i + 1, j]]))
        # West
        if i == 0 or not bin_matrix[i - 1, j]:
            edges.append(np.array([[i, j], [i, j + 1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine the edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:
        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)
        loop.pop()
        loop_list.append(np.array(loop))

    return loop_list


def upscale_matrix_factor(matrix, upscale_factor):
    return np.kron(matrix, np.ones((upscale_factor, upscale_factor)))


def matrix_point8_connectivity(
    matrix,
    connect: Literal["connect", "disconnect"] = "connect",
):
    block1 = np.array([[0, 1], [1, 0]])
    block2 = np.array([[1, 0], [0, 1]])
    block3 = np.ones(dtype=int, shape=(2, 2)) * (1 if connect == "connect" else 0)

    rows, cols = matrix.shape

    # Create a copy of the matrix to store the modified version
    modified_matrix = np.copy(matrix)

    # Iterate over each position in the matrix
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Extract the current 2x2 block
            current_block = matrix[i : i + 2, j : j + 2]

            # Compare the current block with block1 and block2
            if np.array_equal(current_block, block1) or np.array_equal(
                current_block, block2
            ):
                # Replace the current block with block3
                modified_matrix[i : i + 2, j : j + 2] = block3

    return modified_matrix


def matrix2shapes(bin_matrix):
    edges = get_all_edges(bin_matrix=bin_matrix)
    outlines = close_loop_edges(edges=edges)
    # print(np.shape(outlines), outlines)
    return outlines


def translateRotate(bin_matrix, arr):
    arr[:, [0, 1]] = arr[:, [1, 0]]
    arr[:, 1] = arr[:, 1] * (-1) + np.shape(bin_matrix)[0]
    return arr


def matrix_fold4symmetry(x_ini):
    x1 = np.flip(x_ini, axis=0)
    x2 = np.flip(x_ini, axis=1)
    x3 = np.flip(x2, axis=0)
    return np.hstack([np.vstack([x2, x3]), np.vstack([x_ini, x1])])


def polygon_fold4symmetry(polygons_lst, labels_lst):
    polygons_lst = (
        polygons_lst
        + [np.column_stack((poly[:, 0] * -1, poly[:, 1])) for poly in polygons_lst]
        + [np.column_stack((poly[:, 0], poly[:, 1] * -1)) for poly in polygons_lst]
        + [poly * -1 for poly in polygons_lst]
    )
    return polygons_lst, labels_lst * 4


def cut_coners_poly(polygon):
    """ """
    dd = 1
    if len(polygon) < 6:
        return polygon
    mask1 = np.array([[dd, 0], [0, -dd], [dd, 0], [0, -dd]])
    mask2 = np.array([[-dd, 0], [0, dd], [-dd, 0], [0, dd]])
    mask3 = np.array([[-dd, 0], [0, -dd], [-dd, 0], [0, -dd]])
    mask4 = np.array([[-dd, 0], [-dd, 0], [0, -dd], [-dd, 0]])
    mask5 = np.array([[0, dd], [dd, 0], [0, dd], [dd, 0]])
    mask6 = np.array([[0, dd], [dd, 0], [0, dd], [0, dd]])
    mask7 = np.array([[0, dd], [dd, 0], [dd, 0], [0, dd]])
    mask8 = np.array([[0, dd], [-dd, 0], [0, dd], [0, dd]])
    mask9 = np.array([[0, -dd], [-dd, 0], [0, -dd], [0, -dd]])
    mask10 = np.array([[0, -dd], [-dd, 0], [0, -dd], [-dd, 0]])
    mask11 = np.array([[0, dd], [-dd, 0], [-dd, 0], [0, dd]])
    mask12 = np.array([[dd, 0], [0, -dd], [dd, 0], [dd, 0]])
    polygon_roll = np.roll(polygon, shift=-1, axis=0)
    delta = polygon_roll - polygon
    delta[delta < 0] = -1
    delta[delta > 0] = 1
    delta = np.vstack([delta[-1], delta])
    # print(np.hstack([polygon, polygon_roll]), "\n")
    if delta[0, 0] == -dd or delta[0, 1] == -dd:
        polygon = np.flip(polygon, axis=0)
        delta = np.flip(delta * -1, axis=0)
    # print(delta, "\n")

    polygon = np.vstack([polygon[-1], polygon])

    # Initialize an empty boolean matrix with the same shape as the input array
    point_mask = np.ones(delta.shape, dtype=bool)
    rows, cols = delta.shape

    # Iterate over each position in the matrix
    for i in range(rows - 1):
        # Extract the current 2x2 block
        current_block = delta[i : i + 4, 0:2]

        # Compare the current block with block1 and block2
        if (
            np.array_equal(current_block, mask1)
            or np.array_equal(current_block, mask2)
            or np.array_equal(current_block, mask3)
            or np.array_equal(current_block, mask4)
            or np.array_equal(current_block, mask5)
            or np.array_equal(current_block, mask6)
            or np.array_equal(current_block, mask7)
            or np.array_equal(current_block, mask8)
            or np.array_equal(current_block, mask9)
            or np.array_equal(current_block, mask10)
            or np.array_equal(current_block, mask11)
            or np.array_equal(current_block, mask12)
        ):
            point_mask[i + 1 : i + 3, 0:2] = False

    return np.reshape(polygon[point_mask], (-1, 2))


def find_feed_mount_point(polygons, polygonLabels):
    distance_lst = []
    for poly, label in zip(polygons, polygonLabels):
        if label == 1:
            x_mean = np.sum(poly[:, 0]) / poly[:, 0].shape[0]
            y_mean = np.sum(poly[:, 1]) / poly[:, 1].shape[0]
            slope = x_mean / y_mean
            if slope <= 1.1:
                weight = 0.4
                distance = np.sqrt(x_mean**2 + weight * y_mean**2)
                distance_lst.append(distance)
            else:
                distance_lst.append(np.inf)
        else:
            distance_lst.append(np.inf)
    idx_closest = np.argmin(np.array(distance_lst))
    mount_poly = polygons[idx_closest]

    idx_min_oy = np.where(mount_poly[:, 1] == mount_poly[:, 1].min())[0]
    idx_max_ox = np.where(
        mount_poly[:, 0][idx_min_oy] == mount_poly[:, 0][idx_min_oy].max()
    )[0]

    mount_point1 = mount_poly[idx_min_oy][idx_max_ox][0]

    # find min(x) min(y) point
    idx_min_ox = np.where(mount_poly[:, 0] == mount_poly[:, 0].min())[0]
    idx_min_oy = np.where(
        mount_poly[:, 1][idx_min_ox] == mount_poly[:, 1][idx_min_ox].min()
    )[0]
    mount_point2 = mount_poly[idx_min_ox][idx_min_oy][0]

    return mount_point1, mount_point2


def matrix2Holes_n_Polygons(
    bin_matrix,
    lengthPerPixel,
    upscale_factor,
    point8_connectivity: Literal["connect", "disconnect"] = "connect",
    fold4symmetry=False,
    fold4symmetry_poly=False,
    cut_coners_filter=False,
    centering=False,
    feed_cfg=(None, None),
    xc=None,
    yc=None,
):
    """
    Parses a binary matrix into polygons and holes based on connectivity.

    Parameters:
        bin_matrix (np.ndarray): Binary matrix input.
        lengthPerPixel (float): Scaling factor for polygons.
        upscale_factor (int): Factor to upscale the matrix.
        point8_connectivity (Literal): "connect" or "disconnect" for point connectivity.
        fold4symmetry (bool): If True, applies 4-fold symmetry to the matrix.
        fold4symmetry_poly (bool): If True, applies 4-fold symmetry to polygons.
        cut_coners_filter (bool): If True, applies corner cutting filter to polygons.
        centering (bool): If True, centers the polygons.
        feed_cfg (tuple): Configuration for feed points, (None, feedDiam).
        xc, yc (float): Coordinates for centering.

    Returns:
        polygons_lst (list): List of polygons.
        all_labels_lst (list): List of labels (1 for polygon, 0 for hole).
        xc, yc (float): Center coordinates.
        bin_matrix (np.ndarray): Processed binary matrix.
    """

    # Upscale
    if upscale_factor and upscale_factor > 1:
        bin_matrix = upscale_matrix_factor(bin_matrix, upscale_factor=upscale_factor)

    # Apply Connectivity
    if point8_connectivity in ["connect", "disconnect"]:
        bin_matrix = matrix_point8_connectivity(bin_matrix, connect=point8_connectivity)

    # Apply 4-Fold Symmetry
    if fold4symmetry:
        bin_matrix = matrix_fold4symmetry(bin_matrix)

    # Label Islands
    arrIslands, num = measure.label(bin_matrix, connectivity=1, return_num=True)

    # Create Layers of Connectivity
    bin_layers = [(arrIslands == i + 1) for i in range(num)]

    # Initialize Output Lists
    all_labels_lst = []
    polygons_lst = []

    for bin_layer in bin_layers:
        polygonLayer = matrix2shapes(bin_layer)

        # Translate, Rotate, and Scale Polygons
        polygonLayer = [
            translateRotate(bin_matrix, poly) * lengthPerPixel for poly in polygonLayer
        ]

        # Differentiate Holes vs. Polygons
        if len(polygonLayer) > 1:
            area0 = 0
            labels = []
            for poly in polygonLayer:
                oxlen = np.ptp(poly[:, 0])
                oylen = np.ptp(poly[:, 1])
                area = oxlen * oylen
                if area > area0:
                    labels = [0] * len(labels) + [1]
                    area0 = area
                else:
                    labels.append(0)
                polygons_lst.append(poly)
        else:
            labels = [1]
            polygons_lst.append(polygonLayer[0])

        all_labels_lst.extend(labels)

    # Center Polygons
    if centering:
        xc = xc or (bin_matrix.shape[1] * lengthPerPixel / 2)
        yc = yc or (bin_matrix.shape[0] * lengthPerPixel / 2)
        for poly in polygons_lst:
            poly[:, 0] = (poly[:, 0] - xc) / upscale_factor
            poly[:, 1] = (poly[:, 1] - yc) / upscale_factor
    xc, yc = 0.0, 0.0

    # Apply Corner Cutting Filter
    if cut_coners_filter:
        assert (
            not fold4symmetry
        ), "fold4symmetry and cut_coners_filter cannot be used together."
        polygons_lst = [cut_coners_poly(poly) for poly in polygons_lst]

    # Add Feed Polygon
    if feed_cfg[0]:
        mount_point1, mount_point2 = find_feed_mount_point(polygons_lst, all_labels_lst)
        feed_length = feed_cfg[1]
        feed_poly = np.array(
            [
                [0, 0],
                [feed_length / 2, 0],
                [mount_point1[0], mount_point1[1]],
                [mount_point2[0], mount_point2[1]],
            ]
        )
        polygons_lst.insert(0, feed_poly)
        all_labels_lst.insert(0, 1)

    # Apply 4-Fold Symmetry to Polygons
    if fold4symmetry_poly:
        polygons_lst, all_labels_lst = polygon_fold4symmetry(
            polygons_lst, all_labels_lst
        )

    return polygons_lst, all_labels_lst, xc, yc, bin_matrix


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
        bin_matrix: np.ndarray,
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
        save_mat: bool = True,
        save_npy: bool = False,
    ):
        self.bin_matrix = bin_matrix
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

        # Set polygon space and plot sizes
        scale_factor = 2 if fold4symmetry or fold4symmetry_poly else 1
        self.poly_space_size = (
            self.bin_matrix.shape[0] * scale_factor * lengthPerPixel,
            self.bin_matrix.shape[1] * scale_factor * lengthPerPixel,
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
            self.bin_matrix_final,
        ) = self.init_polygons()
        self.polygon_stack = self.init_polygon_stack()

        # Save outputs
        if save_mat:
            self.polygon_stack_path = self.save_polygon_stack()
        if save_npy:
            self.save_bin_matrix_as_npy()

        self.coverage = count_coverage(self.bin_matrix_final)

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
        npy_file_name = (
            f"bin_matrixDebug{self.bin_matrix_final.shape[0]}x{self.bin_matrix_final.shape[1]}_iter{self.generation_idx}.npy"
        )
        npy_path = Path(sub_dir, npy_file_name)
        np.save(npy_path, self.bin_matrix_final)

    def save_polygon_stack(self):
        """Save the polygon stack as a .mat file."""
        sub_dir = Path(self.save_dir, "mat")
        sub_dir.mkdir(parents=True, exist_ok=True)
        mat_file_name = (
            f"Poly{self.bin_matrix_final.shape[0]}x{self.bin_matrix_final.shape[1]}_iter{self.generation_idx}.mat"
        )
        polygon_stack_path = Path(sub_dir, mat_file_name)
        savemat(polygon_stack_path, {"data": self.polygon_stack})
        print(f"Polygon stack saved to: {polygon_stack_path}")
        return polygon_stack_path

    def init_polygons(self):
        """Initialize polygons and related attributes."""
        return matrix2Holes_n_Polygons(
            self.bin_matrix,
            lengthPerPixel=self.lengthPerPixel,
            upscale_factor=self.upscale_factor,
            point8_connectivity=self.point8_connectivity,
            fold4symmetry=self.fold4symmetry,
            fold4symmetry_poly=self.fold4symmetry_poly,
            cut_coners_filter=self.cut_coners_filter,
            centering=self.centering,
            feed_cfg=self.feed_cfg,
        )

    def init_polygon_stack(self):
        """Initialize the polygon stack."""
        polygon_stack = np.empty((len(self.polygons), 2), dtype=object)
        for i, (polygon, label) in enumerate(zip(self.polygons, self.polygonLabels)):
            polygon_stack[i] = np.array([[polygon, label]], dtype=object)
        return polygon_stack

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
            "plasma": plt.cm.plasma,
            "inferno": plt.cm.inferno,
            "viridis": plt.cm.viridis,
            "rainbow": plt.cm.rainbow,
        }
        selected_colormap = colormaps.get(colormap, plt.cm.rainbow)
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
            border = plt.Rectangle(
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

        png_name = (
            f"Poly{self.bin_matrix_final.shape[0]}x{self.bin_matrix_final.shape[1]}_epoch{self.generation_idx:04d}.png"
        )
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
