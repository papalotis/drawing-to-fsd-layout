import networkx as nx
import numpy as np
import streamlit as st
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

from drawing_to_fsd_layout.common import FloatArray2, FloatArrayNx2, IntArrayN


def circle_fit(coords: np.ndarray, max_iter: int = 99) -> np.ndarray:
    """
    Function taken from: https://github.com/papalotis/ft-fsd-path-planning/blob/d82ba8f93c753a9d0fe0c77fa8c9af88aafad6ea/fsd_path_planning/utils/math_utils.py#L569

    Fit a circle to a set of points. This function is adapted from the hyper_fit
    function in the circle-fit package (https://pypi.org/project/circle-fit/).
    The function is a njit version of the original function with some input validation
    removed. Furthermore, the residuals are not calculated or returned.
    Args:
        coords: The coordinates of the points as an [N, 2] array.
        max_iter: The maximum number of iterations.
    Returns:
        An array with 3 elements:
        - center x
        - center y
        - radius
    """

    X = coords[:, 0]
    Y = coords[:, 1]

    n = X.shape[0]

    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi * Xi + Yi * Yi

    # compute moments
    Mxy = (Xi * Yi).sum() / n
    Mxx = (Xi * Xi).sum() / n
    Myy = (Yi * Yi).sum() / n
    Mxz = (Xi * Zi).sum() / n
    Myz = (Yi * Zi).sum() / n
    Mzz = (Zi * Zi).sum() / n

    # computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz

    A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
    A22 = A2 + A2

    # finding the root of the characteristic polynomial
    y = A0
    x = 0.0
    for _ in range(max_iter):
        Dy = A1 + x * (A22 + 16.0 * x * x)
        x_new = x - y / Dy
        if x_new == x or not np.isfinite(x_new):
            break
        y_new = A0 + x_new * (A1 + x_new * (A2 + 4.0 * x_new * x_new))
        if abs(y_new) >= abs(y):
            break
        x, y = x_new, y_new

    det = x * x - x * Mz + Cov_xy
    X_center = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.0
    Y_center = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.0

    x = X_center + X.mean()
    y = Y_center + Y.mean()
    r = np.sqrt(abs(X_center**2 + Y_center**2 + Mz))

    return np.array([x, y, r])


def create_cyclic_sliding_window_indices(
    window_size: int, step_size: int, signal_length: int
) -> IntArrayN:
    """
    Function taken from https://github.com/papalotis/ft-fsd-path-planning/blob/main/fsd_path_planning/calculate_path/path_parameterization.py
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    half_window_size = window_size // 2

    indexer = (
        np.arange(-half_window_size, half_window_size + 1)
        + np.arange(0, signal_length, step_size).reshape(-1, 1)
    ) % signal_length
    return indexer


def calculate_path_curvature(
    path: FloatArrayNx2, window_size: int, path_is_closed: bool
) -> FloatArrayNx2:
    """
    Function taken from https://github.com/papalotis/ft-fsd-path-planning/blob/main/fsd_path_planning/calculate_path/path_parameterization.py

    Calculate the curvature of the path.
    Args:
        path: The path as a 2D array of points.
        window_size: The size of the window to use for the curvature calculation.
        path_is_closed: Whether the path is closed or not.
    Returns:
        The curvature of the path.
    """
    windows = create_cyclic_sliding_window_indices(
        window_size=window_size, step_size=1, signal_length=len(path)
    )

    path_curvature = np.zeros(len(path))
    for i, window in enumerate(windows):
        if not path_is_closed:
            diff = window[1:] - window[:-1]
            if np.any(diff != 1):
                idx_cutoff = int(np.argmax(diff != 1) + 1)
                if i < window_size:
                    window = window[idx_cutoff:]
                else:
                    window = window[:idx_cutoff]

        points_in_window = path[window]

        _, _, radius = circle_fit(points_in_window)
        radius = min(
            max(radius, 1.0), 3000.0
        )  # np.clip didn't work for some reason (numba bug?)
        curvature = 1 / radius
        three_idxs = np.array([0, int(len(points_in_window) / 2), -1])
        three_points = points_in_window[three_idxs]
        hom_points = np.column_stack((np.ones(3), three_points))
        sign = np.linalg.det(hom_points)

        path_curvature[i] = curvature * np.sign(sign)

    mode = "wrap" if path_is_closed else "nearest"

    filtered_curvature = uniform_filter1d(
        path_curvature, size=window_size // 10, mode=mode
    )
    # filtered_curvature = path_curvature

    return filtered_curvature


def calculate_edge_curvature(edge: FloatArrayNx2) -> FloatArrayNx2:
    window_size = len(edge) // 25
    if window_size % 2 == 0:
        window_size += 1
    return calculate_path_curvature(edge, window_size=window_size, path_is_closed=True)


def place_cones_for_edge(edge: FloatArrayNx2) -> FloatArrayNx2:
    """
    Place cones along an edge.

    Args:
        edge: The edge to place cones on. Shape (n,2)
        cone_spacing: The distance between cones.

    Returns:
        The locations of the cones. Shape (m,2)
    """
    raise NotImplementedError


def calculate_min_track_width_index(
    edge_1: FloatArrayNx2, edge_2: FloatArrayNx2
) -> tuple[int, int]:
    distances = cdist(edge_1, edge_2)
    idx_min_distance_to_1 = np.min(distances, axis=1).argmin()
    idx_min_distance_to_2 = np.min(distances, axis=0).argmin()

    return idx_min_distance_to_1, idx_min_distance_to_2


def calculate_min_track_width(edge_1: FloatArrayNx2, edge_2: FloatArrayNx2) -> float:
    idx_min_distance_to_1, idx_min_distance_to_2 = calculate_min_track_width_index(
        edge_1, edge_2
    )
    edge_1_closest_point = edge_1[idx_min_distance_to_1]
    edge_2_closest_point = edge_2[idx_min_distance_to_2]

    return np.linalg.norm(edge_1_closest_point - edge_2_closest_point)


def estimate_centerline_from_edges(
    edge_1: FloatArrayNx2, edge_2: FloatArrayNx2
) -> FloatArrayNx2:
    shorter, longer = sorted([edge_1, edge_2], key=len)
    idx_shorter = cdist(shorter, longer).argmin(axis=0)
    shorter_sampled = shorter[idx_shorter]
    centerline = (longer + shorter_sampled) / 2

    return centerline


def estimate_centerline_length(edge_1: FloatArrayNx2, edge_2: FloatArrayNx2) -> float:
    edge_1_sampled = edge_1[::10]
    edge_2_sampled = edge_2[::10]
    centerline = estimate_centerline_from_edges(edge_1_sampled, edge_2_sampled)
    return np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))


def split_edge_to_straight_and_curve(
    edge: FloatArrayNx2,
    curvature_threshold_upper: float = 0.02,
    curvature_threshold_lower: float = 0.01,
) -> IntArrayN:
    """
    Split an edge into straight and curve sections.
    """
    curvature_threshold_lower, curvature_threshold_upper = sorted(
        (curvature_threshold_lower, curvature_threshold_upper)
    )
    curvature = calculate_edge_curvature(edge)
    start_index = np.argmin(curvature)  # start from a straight

    # 0 is straight, 1 is left curve, -1 is right curve
    out = np.zeros(len(curvature), dtype=int)
    out[start_index] = 0
    for offset in range(1, len(curvature) + 1):
        previous_index = (start_index + offset - 1) % len(curvature)
        next_index = (start_index + offset) % len(curvature)

        state = out[previous_index]
        if state == 0 and abs(curvature[next_index]) > curvature_threshold_upper:
            state = np.sign(curvature[next_index])
        elif state != 0 and abs(curvature[next_index]) < curvature_threshold_lower:
            state = 0

        out[next_index] = state

    return out


def decide_start_finish_line_position_and_direction(
    edge: FloatArrayNx2,
) -> tuple[FloatArray2, FloatArray2]:
    track_point_types = split_edge_to_straight_and_curve(edge)
    straights_indices: list[list[int]] = [[]]
    for i, track_point_type in enumerate(track_point_types):
        if track_point_type == 0:
            straights_indices[-1].append(i)
        else:
            if len(straights_indices[-1]) > 0:
                straights_indices.append([])

    longest_straight_indices = max(straights_indices, key=len)
    start_index = longest_straight_indices[len(longest_straight_indices) // 2]
    direction_of_straight = np.diff(edge[longest_straight_indices], axis=0).mean(axis=0)
    return edge[start_index], direction_of_straight / np.linalg.norm(
        direction_of_straight
    )


def fix_edge_direction(
    edge: FloatArrayNx2, start_position: FloatArray2, start_direction: FloatArray2
) -> FloatArrayNx2:
    idx_edge_start = np.linalg.norm(edge - start_position, axis=1).argmin()
    edge_rolled = np.roll(edge, -idx_edge_start, axis=0)

    edge_direction = edge_rolled[2] - edge_rolled[0]
    dot = np.dot(edge_direction, start_direction)
    if dot < 0:
        edge_rolled = edge_rolled[::-1]

    return edge_rolled


def place_cones(
    trace: FloatArrayNx2, seed: int, mean: float, variance: float
) -> FloatArrayNx2:
    rng = np.random.default_rng(seed)

    idxs_to_keep = [0]
    next_distance = rng.normal(mean, variance)
    next_idx = idxs_to_keep[0]
    while next_idx < len(trace):
        trace_from_last_in = trace[next_idx:]
        dist_to_next = np.linalg.norm(np.diff(trace_from_last_in, axis=0), axis=1)
        cum_dist = np.cumsum(dist_to_next)

        offset_next_idx = np.argmax(cum_dist > next_distance)

        if offset_next_idx == 0:
            break

        next_idx = offset_next_idx + next_idx
        idxs_to_keep.append(next_idx)
        next_distance = rng.normal(mean, variance)

    randomly_placed_cones = trace[idxs_to_keep]

    st.write(np.linalg.norm(np.diff(randomly_placed_cones, axis=0), axis=1).mean())


    return randomly_placed_cones
