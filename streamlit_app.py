from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage import io

from drawing_to_fsd_layout.common import FloatArrayNx2
from drawing_to_fsd_layout.cone_placement import (
    calculate_min_track_width, calculate_min_track_width_index,
    decide_start_finish_line_position_and_direction,
    estimate_centerline_from_edges, estimate_centerline_length,
    fix_edge_direction, place_cones, split_edge_to_straight_and_curve)
from drawing_to_fsd_layout.export import cones_to_lyt, export_json_string
from drawing_to_fsd_layout.image_processing import (
    extract_track_edges, fix_edges_orientation_and_scale_to_unit,
    load_image_and_preprocess, rotate)
from drawing_to_fsd_layout.spline_fit import SplineFitterFactory


class UploadType(str, Enum):
    FILE = "file"
    URL = "url"


class ScalingMethod(str, Enum):
    MIN_TRACK_WIDTH = "min_track_width"
    CENTERLINE_LENGTH = "centerline_length"


class SFLineCalculationMethod(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class SmoothingDegree(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


SFLineHelpText = """
The start/finish line is the line that the car crosses to start and finish a lap.
The start/finish line is calculated automatically by default. If you want to
manually specify the start/finish line, select "Manual" and enter the x/y coordinates
of the start/finish line as well as its direction. The closest point on the track
to your assigned start/finish line will be used.
"""


def image_upload_widget() -> np.ndarray:
    upload_type = UploadType[st.radio("Upload type", [x.name for x in UploadType])]

    if upload_type == UploadType.FILE:
        uploaded_file = st.file_uploader("Upload an image")
        if uploaded_file is None:
            st.info("Please upload an image")
            st.stop()

        imread_input = uploaded_file.read()
        kwargs_imageio = dict(plugin="imageio")
    elif upload_type == UploadType.URL:
        imread_input = st.text_input("URL to image")
        if imread_input == "":
            st.info("Please enter a URL")
            st.stop()

        kwargs_imageio = dict()
    else:
        raise AssertionError("Unreachable code")

    image = io.imread(imread_input, **kwargs_imageio)

    return image


def plot_contours(
    contour_a: FloatArrayNx2, contour_b: FloatArrayNx2, do_show: bool = True
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    for con in (contour_a, contour_b):
        plt.plot(*con.T, "-")
    plt.axis("equal")
    if do_show:
        st.pyplot(plt.gcf())


def main() -> None:
    st.title("Upload image")
    image = image_upload_widget()
    st.image(image, caption="Uploaded image")
    with st.spinner("Preprocessing image"):
        preprocessed_image = load_image_and_preprocess(image)

    with st.spinner("Extracting track edges"):
        (
            contour_a,
            contour_b,
        ) = extract_track_edges(preprocessed_image)

    contour_a_fixed, contour_b_fixed = fix_edges_orientation_and_scale_to_unit(
        contour_a, contour_b
    )

    st.title("Scale to real-world units")
    scaling_method = st.radio(
        "Method to use to scale to real world units",
        list(ScalingMethod),
        format_func=lambda x: x.name.replace("_", " ").title(),
        horizontal=True,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        desired_track_width = st.number_input(
            "Min track width",
            min_value=0.1,
            value=3.0,
            disabled=scaling_method != ScalingMethod.MIN_TRACK_WIDTH,
        )
    with col_right:
        desired_centerline_length = st.number_input(
            "Centerline length",
            min_value=1.0,
            value=200.0,
            disabled=scaling_method != ScalingMethod.CENTERLINE_LENGTH,
        )

    if scaling_method == ScalingMethod.MIN_TRACK_WIDTH:
        unscaled_min_track_width = calculate_min_track_width(
            contour_a_fixed, contour_b_fixed
        )
        scale = desired_track_width / unscaled_min_track_width
    elif scaling_method == ScalingMethod.CENTERLINE_LENGTH:
        unscaled_centerline_length = estimate_centerline_length(
            contour_a_fixed, contour_b_fixed
        )
        scale = desired_centerline_length / unscaled_centerline_length

    else:
        raise AssertionError("Unreachable code")

    contour_a_fixed_scaled = contour_a_fixed * scale
    contour_b_fixed_scaled = contour_b_fixed * scale

    smoothing_degree = st.select_slider(
        "Smoothing",
        options=list(SmoothingDegree),
        value=SmoothingDegree.LIGHT,
        format_func=lambda x: x.name.replace("_", " ").title(),
        help="For hand-drawn tracks it is useful to smooth the track edges.",
    )
    st.write(scale)
    smoothing = scale / 5 * list(SmoothingDegree).index(smoothing_degree)
    st.write("before spline")
    spline_factory = SplineFitterFactory(
        smoothing=smoothing, predict_every=0.3, max_deg=3
    )

    contour_a_splined = spline_factory.fit(
        contour_a_fixed_scaled[::2], periodic=True
    ).predict(der=0)

    st.write(len(contour_a_splined), len(contour_a_fixed_scaled))

    contour_b_splined = spline_factory.fit(
        contour_b_fixed_scaled[::2], periodic=True
    ).predict(der=0)

    min_track_width = calculate_min_track_width(contour_a_splined, contour_b_splined)
    st.write(min_track_width)
    centerline_length = estimate_centerline_length(contour_a_splined, contour_b_splined)
    st.write(centerline_length)

    st.markdown(
        f"*Min track width: {min_track_width:.2f} m | Centerline length:"
        f" {centerline_length:.2f} m*"
    )

    # centerline = estimate_centerline_from_edges(contour_a_splined, contour_b_splined)

    st.title("Place start/finish line")
    st.write("hi")
    sf_line_calculation_method = SFLineCalculationMethod(
        st.radio(
            "Method to use to place start/finish line",
            list(SFLineCalculationMethod),
            format_func=lambda x: x.name.replace("_", " ").title(),
            horizontal=True,
            help=SFLineHelpText,
        )
    )
    (
        start_on_contour_a,
        direction_start,
    ) = decide_start_finish_line_position_and_direction(contour_a_splined)
    st.write("after decide_start_finish_line_position_and_direction")
    if sf_line_calculation_method == SFLineCalculationMethod.AUTO:
        flip_direction = st.checkbox(
            "Flip track direction",
            value=False,
        )
        if flip_direction:
            direction_start = -direction_start
    elif sf_line_calculation_method == SFLineCalculationMethod.MANUAL:
        col_left, col_middle, col_right = st.columns(3)
        with col_left:
            start_finish_line_x = st.number_input(
                "S/F line x position (x)", value=start_on_contour_a[0]
            )

        with col_middle:
            start_finish_line_y = st.number_input(
                "S/F line y position (m)", value=start_on_contour_a[1]
            )

        with col_right:
            start_finish_line_angle = st.number_input(
                "S/F line angle (deg)",
                value=np.rad2deg(np.arctan2(*direction_start[::-1])),
            )
        user_point = np.array([start_finish_line_x, start_finish_line_y])
        user_direction = np.array(
            [
                np.cos(np.deg2rad(start_finish_line_angle)),
                np.sin(np.deg2rad(start_finish_line_angle)),
            ]
        )
        from scipy.spatial.distance import cdist

        start_on_contour_a = contour_a_splined[
            cdist([user_point], contour_a_splined).argmin()
        ]
        direction_start = user_direction
    from scipy.spatial.distance import cdist

    index_start_contour_b = cdist([start_on_contour_a], contour_b_fixed_scaled).argmin()
    start_on_contour_b = contour_b_fixed_scaled[index_start_contour_b]

    direction_to_inside_of_track = start_on_contour_b - start_on_contour_a
    direction_to_outside_of_track = -direction_to_inside_of_track / np.linalg.norm(
        direction_to_inside_of_track
    )
    direction_to_outside_of_track *= min_track_width * 2

    start_finish_arrow_stem = start_on_contour_a + direction_to_outside_of_track

    contour_a_final = fix_edge_direction(
        contour_a_splined, start_on_contour_a, direction_start
    )
    contour_b_final = fix_edge_direction(
        contour_b_splined, start_on_contour_b, direction_start
    )

    cones_a = place_cones(contour_a_final, 1, 4, 0.1)
    cones_b = place_cones(contour_b_final, 1, 4, 0.1)

    start_middle = (cones_a[0] + cones_b[0]) / 2
    start_direction_a = cones_a[1] - cones_a[0]
    start_direction_a /= np.linalg.norm(start_direction_a)

    start_direction_b = cones_b[1] - cones_b[0]
    start_direction_b /= np.linalg.norm(start_direction_b)

    start_direction = (start_direction_a + start_direction_b) / 2

    cones_a_trans = cones_a - start_middle
    cones_a_trans_rot = rotate(cones_a_trans, -np.arctan2(*start_direction[::-1]))
    if cones_a_trans_rot[0, 1] > 0:
        left_cones = cones_a
        right_cones = cones_b
    else:
        left_cones = cones_b
        right_cones = cones_a

    plt.figure(figsize=(10, 10))

    plt.arrow(
        *start_finish_arrow_stem,
        *direction_start * min_track_width * 2,
        head_width=min_track_width,
    )

    plt.plot(*right_cones.T, ".", c="gold", label="Contour A", markersize=2)
    plt.plot(*left_cones.T, ".", c="b", label="Contour B", markersize=2)
    plt.plot(
        [left_cones[0, 0], right_cones[0, 0]],
        [left_cones[0, 1], right_cones[0, 1]],
        "x",
        c="orange",
    )
    plt.axis("equal")
    plt.title("Final track layout")

    st.pyplot(plt.gcf())

    st.title("Export")
    track_name = st.text_input("Track name", "Custom Track")
    track_name_normalized = track_name.replace(" ", "_").lower()

    tab_json, tab_lfs = st.tabs(["JSON", "Live for Speed Layout"])

    with tab_json:
        json_string = export_json_string(left_cones, right_cones)
        st.download_button(
            "Download JSON",
            json_string,
            file_name=f"{track_name_normalized}.json",
            mime="application/json",
        )
    with tab_lfs:
        world_name = "LA2"
        lyt_bytes = cones_to_lyt(world_name, left_cones, right_cones)

        filename = f"{world_name}_{track_name_normalized}.lyt"

        st.download_button(
            "Download LFS Layout",
            lyt_bytes,
            file_name=filename,
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
