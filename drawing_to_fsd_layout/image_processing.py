from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
from scipy.spatial.distance import cdist
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import unsharp_mask
from skimage.transform import rescale

from drawing_to_fsd_layout.common import FloatArrayNx2

Image = np.typing.NDArray[np.float64]


def rotate(points: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates the points in `points` by angle `theta` around the origin

    Args:
        points: The points to rotate. Shape (n,2)
        theta: The angle by which to rotate in radians

    Returns:
        The points rotated
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((cos_theta, -sin_theta), (sin_theta, cos_theta))).T
    return np.dot(points, rotation_matrix)


def load_raw_image(image_path: Path | str | Image) -> Image:
    """
    Load an image. If input is a string, it is assumed to be a path to an image. If
    input is a Path, it is assumed to be a path to an image. If input is an image, it is
    returned as is.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)

    image = io.imread(image_path) if isinstance(image_path, Path) else image_path.copy()

    return image


def create_close_point_adjacency_matrix(
    edge_1: FloatArrayNx2, edge_2: FloatArrayNx2, distance: float
) -> np.ndarray:
    """
    Combine two edges into a single edge.

    Args:
        edge_1: The first edge. Shape (n,2)
        edge_2: The second edge. Shape (m,2)

    Returns:
        The combined edge.
    """
    dist = cdist(edge_1, edge_2)

    # find the closest point in edge_2 for each point in edge_1
    idxs_closest_in_edge_2 = dist.argmin(axis=1)

    # find the closest point in edge_1 for each point in edge_2
    idxs_closest_in_edge_1 = dist.argmin(axis=0)

    adj_edge_1_to_edge_2 = idxs_closest_in_edge_1 < distance
    adj_edge_2_to_edge_1 = idxs_closest_in_edge_2 < distance

    combined_length = len(edge_1) + len(edge_2)

    adj = np.zeros((combined_length, combined_length), dtype=bool)

    adj[: len(edge_1), len(edge_1) :] = adj_edge_1_to_edge_2
    adj[len(edge_1) :, : len(edge_1)] = adj_edge_2_to_edge_1

    return adj


def combine_edges(edge_1: FloatArrayNx2, edge_2: FloatArrayNx2) -> FloatArrayNx2:
    """
    Combine two edges into a single edge.

    Args:
        edge_1: The first edge. Shape (n,2)
        edge_2: The second edge. Shape (m,2)

    Returns:
        The combined edge.
    """
    adj = create_close_point_adjacency_matrix(edge_1, edge_2, distance=2)
    # remove one directional edges
    adj = np.logical_and(adj, adj.T)

    # find the index of the closest point in edge_2 for each point in edge_1
    idx_edge_2_to_edge_1_pair = np.argmax(adj[: len(edge_1), len(edge_1) :], axis=1)
    # if argmax returns 0, it means that there is no edge between the two points
    filtered_idxs_mask = idx_edge_2_to_edge_1_pair != 0

    edge_2_keep_idxs = idx_edge_2_to_edge_1_pair[filtered_idxs_mask]
    edge_1_keep_idxs = np.arange(len(edge_1))[filtered_idxs_mask]

    # combine the edges by element-wise mean
    edge_1_keep = edge_1[edge_1_keep_idxs]
    edge_2_keep = edge_2[edge_2_keep_idxs]
    combined_edge = (edge_1_keep + edge_2_keep) / 2

    return combined_edge


@st.cache(show_spinner=False)
def load_image_and_preprocess(
    image_path: Path | str | Image, target_size: tuple[int, int] = (1000, 1000)
) -> Image:
    """Load an image and scale it to the correct size for FSD layout."""
    image = load_raw_image(image_path)

    if image.ndim > 3:
        raise ValueError(f"Image must be 2D or 3D. Image has {image.ndim} dimensions.")

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = rgb2gray(image)
        else:
            image = np.mean(image[:, :, :3], axis=-1)

    if image.ndim == 3:
        image = image[:, :, 0]

    image_resolution = np.prod(image.shape)
    target_resolution = np.prod(target_size)

    rescale_ratio = target_resolution / image_resolution
    # we need to take the root of the ratio to achieve the correct scaling
    image_resized = rescale(image, rescale_ratio**0.5)

    # apply unsharp mask
    image_unsharp = unsharp_mask(image_resized, radius=5, amount=3)

    return image_unsharp


@st.cache(show_spinner=False)
def reorder_track_edge(
    g: nx.Graph, cc_idxs: Iterable[int], all_nodes_positions: FloatArrayNx2
) -> FloatArrayNx2:
    cc_idxs_list = list(cc_idxs)
    start_index = cc_idxs_list[0]

    idxs_ordered = list(nx.depth_first_search.dfs_preorder_nodes(g, start_index))

    cc_positions_ordered = all_nodes_positions[idxs_ordered]

    distances = np.linalg.norm(np.diff(cc_positions_ordered, axis=0), axis=1)
    mask_remove = distances > 5

    first_index_over_k = np.argmax(mask_remove)
    cc_positions_ordered = cc_positions_ordered[:first_index_over_k]

    cc_positions_ordered_low_res = cc_positions_ordered

    while len(cc_positions_ordered_low_res) > 1000:
        cc_positions_ordered_low_res = cc_positions_ordered_low_res[::2]

    return cc_positions_ordered_low_res


@st.cache(show_spinner=False, suppress_st_warning=True)
def extract_track_edges(
    image: Image,
    show_steps: bool = False,
) -> tuple[FloatArrayNx2, FloatArrayNx2]:
    # apply canny edge detection with increasing sigma until the number of pixels
    # designated as edges is low enough
    for s in np.linspace(2, 5, 20):
        image_canny = canny(image, sigma=s).astype(float)
        m = image_canny.mean()
        if m < 0.03:  # prior, found by trial and error
            break

    if show_steps:
        st.image(image_canny, caption="Canny edge detection")

    # treat each pixel as a node in a graph and connect nodes that are close to each
    # other
    edge_pixels = np.argwhere(image_canny > 0.01)
    dist = cdist(edge_pixels, edge_pixels)
    adj = dist < 2
    adj[np.eye(len(adj), len(adj), dtype=bool)] = 0
    g = nx.from_numpy_matrix(adj)

    # find the connected components of the graph
    # we expect four connected components, two for the outer edge and two for the inner
    # edge
    cc = list(nx.connected_components(g))

    best_ccs = sorted(cc, key=len, reverse=True)[:4]

    best_clusters = [edge_pixels[list(idxs_in_cc)] for idxs_in_cc in best_ccs]
    # keep the longer version of each track edge
    if len(best_clusters) == 2:
        outer, inner = best_clusters
        cc_outer, cc_inner = best_ccs
    elif len(best_clusters) == 4:
        outer, _, inner, _ = best_clusters
        cc_outer, _, cc_inner, _ = best_ccs
    elif len(best_clusters) == 1:
        st.error(
            "There was an error extracting the two track edges. Have you drawn a closed track?"
        )
        st.stop()
    elif len(best_clusters) == 0:
        st.error("No track edges were found. Have you drawn a track?")
        st.stop()

    if show_steps:
        plt.figure()
        plt.plot(*outer.T, ".", markersize=1)
        plt.plot(*inner.T, ".", markersize=1)
        plt.axis("equal")
        st.pyplot(plt.gcf())

    outer_ordered = reorder_track_edge(g, cc_outer, edge_pixels)
    inner_ordered = reorder_track_edge(g, cc_inner, edge_pixels)

    if show_steps:
        plt.figure()
        plt.plot(*outer_ordered[::3].T, "-")
        plt.plot(*inner_ordered[::3].T, "-")
        plt.axis("equal")
        st.pyplot(plt.gcf())

    return outer_ordered, inner_ordered


@st.cache(show_spinner=False)
def fix_edges_orientation_and_scale_to_unit(
    edge_a: FloatArrayNx2, edge_b: FloatArrayNx2
) -> tuple[FloatArrayNx2, FloatArrayNx2]:
    points = [edge_a, edge_b]

    all_points = np.concatenate(points)

    n_split = len(points[0])

    ranges = np.ptp(all_points, axis=0)

    index_largest_range = np.argmax(ranges)

    min_value = np.min(all_points[:, index_largest_range])
    range_value = ranges[index_largest_range]

    points_scaled = (all_points - min_value) / range_value

    points_center = points_scaled.mean(axis=0)

    points_centered = points_scaled - points_center
    points_rotated = rotate(points_centered, theta=-3.1415 / 2)

    return points_rotated[:n_split], points_rotated[n_split:]
