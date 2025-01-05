from typing import Literal

import numpy as np
from skimage import measure


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
    rows, _ = delta.shape

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


def matrix_to_polygons(
    bin_matrix,
    lengthPerPixel,
    upscale_factor,
    point8_connectivity: Literal[
        "connect", "disconnect"
    ] = "connect",  # pyright: ignore
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
    arrIslands, num = measure.label(
        bin_matrix, connectivity=1, return_num=True
    )  # pyright: ignore

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
