from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage import io

from drawing_to_fsd_layout.canvas_image import get_canvas_image
from drawing_to_fsd_layout.common import FloatArrayNx2, find_github_link_of_repo
from drawing_to_fsd_layout.cone_placement import (
    calculate_min_track_width,
    decide_start_finish_line_position_and_direction,
    estimate_centerline_length,
    fix_edge_direction,
    place_cones,
)
from drawing_to_fsd_layout.export import (
    cones_to_lyt,
    export_json_string,
)
from drawing_to_fsd_layout.image_processing import (
    extract_track_edges,
    fix_edges_orientation_and_scale_to_unit,
    load_image_and_preprocess,
    rotate,
)
from drawing_to_fsd_layout.spline_fit import SplineFitterFactory

st.set_page_config(page_title="Drawing to Layout", page_icon="🏎️")


class UploadType(str, Enum):
    FILE = "file"
    URL = "url"


class ScalingMethod(str, Enum):
    MINIMUM_TRACK_WIDTH = "minimum_track_width"
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


@st.cache
def load_example_image() -> np.ndarray:
    return io.imread("media/before.png")


def image_upload_widget() -> tuple[np.ndarray, bool]:
    mode = st.radio(
        "Image upload",
        ["Upload", "Canvas", "Example Image"],
        horizontal=True,
        help="Choose how to upload an image. You can upload an image of a track drawing, use the canvas inside this app, or use an example image to get an understanding of how the app works.",
    )

    should_show_image = True

    if mode == "Upload":
        upload_type = UploadType[
            st.radio(
                "Upload type",
                [x.name for x in UploadType],
                horizontal=True,
                help="Choose whether to upload an image file or enter a URL to an image.",
            )
        ]

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

    elif mode == "Example Image":
        image = load_example_image()

    elif mode == "Canvas":
        # the canvas already shows the image, so we don't need to show it again
        should_show_image = False
        image = get_canvas_image()

        # st.image(image)

    assert image is not None
    return image, should_show_image


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
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.image("https://fasttube.de/wp-content/uploads/2016/01/logo_medium_black.png")

    st.title("Drawing to FSD Layout Tool by FaSTTUBe")
    # dynmaically create issues link so that if the repo is forked the link is still correct
    try:
        link = find_github_link_of_repo()
    except ValueError:
        st.write("could not find link to remote")
        link = "https://github.com/papalotis/drawing-to-fsd-layout/"

    if not link.endswith("/"):
        link += "/"

    link += "issues"

    st.warning(
        "This tool is provided as is. It has undergone little testing."
        " There are many bugs and mostly happy path scenarios are considered."
        f" If you find a bug, please report it on the [GitHub repository]({link})."
    )

    st.markdown("## Image")
    image, should_show_image = image_upload_widget()
    if should_show_image:
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

    st.info(
        """Choose how to scale the track to real-world units. You can either specify the minimum track width or the centerline length. The track will be scaled to match the specified value. The scaling is not perfect and the resulting track might not have the exact specified value. You might need to play around with the scaling method and the smoothing to get the desired result.
        
        
The default minimum track width is set to 3.3m. This is because the official minimum track width is 3.0 meters but that is measured from the inside of both track edges. The tool always considers the center of the cones so 0.15 meters are added to each side to compensate. You can change this value to match the track width you want. The track width is the distance between the two track edges.
        """,
        # icon="📏",
    )

    scaling_method = st.radio(
        "Method to use to scale to real world units",
        list(ScalingMethod),
        format_func=lambda x: x.name.replace("_", " ").title(),
        horizontal=True,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        desired_track_width = st.number_input(
            "Minimum track width",
            min_value=2.0,
            max_value=7.0,
            value=3.3,
            disabled=scaling_method != ScalingMethod.MINIMUM_TRACK_WIDTH,
        )
    with col_right:
        desired_centerline_length = st.number_input(
            "Centerline length",
            min_value=200.0,
            max_value=900.0,
            value=350.0,
            disabled=scaling_method != ScalingMethod.CENTERLINE_LENGTH,
        )

    if scaling_method == ScalingMethod.MINIMUM_TRACK_WIDTH:
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
    smoothing = scale / 5 * list(SmoothingDegree).index(smoothing_degree) + 0.1
    spline_factory = SplineFitterFactory(
        smoothing=smoothing, predict_every=0.6, max_deg=3
    )

    contour_a_splined = spline_factory.fit(
        contour_a_fixed_scaled[::3], periodic=True
    ).predict(der=0)

    contour_b_splined = spline_factory.fit(
        contour_b_fixed_scaled[::3], periodic=True
    ).predict(der=0)

    min_track_width = calculate_min_track_width(contour_a_splined, contour_b_splined)
    centerline_length = estimate_centerline_length(contour_a_splined, contour_b_splined)

    st.markdown(
        f"*Min track width: {min_track_width:.2f} m | Centerline length:"
        f" {centerline_length:.2f} m*"
    )

    # centerline = estimate_centerline_from_edges(contour_a_splined, contour_b_splined)

    st.title("Place start/finish line")

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

    plt.plot(
        *right_cones[1:].T,
        "o",
        c="gold",
        label="Contour A",
        markersize=5,
        markeredgecolor="black",
    )
    plt.plot(
        *left_cones[1:].T,
        "o",
        c="b",
        label="Contour B",
        markersize=5,
        markeredgecolor="black",
    )

    plt.plot(
        [left_cones[0, 0], right_cones[0, 0]],
        [left_cones[0, 1], right_cones[0, 1]],
        "o",
        c="orange",
        markersize=7,
        markeredgecolor="black",
    )
    plt.axis("equal")
    plt.title("Final track layout")

    st.pyplot(plt.gcf())
    st.info(
        "The cone markers in the plot are not to scale. They are just for visualization."
    )

    st.title("Export")
    track_name = st.text_input("Track name", "Custom Track")
    track_name_normalized = track_name.replace(" ", "_").lower()

    tab_json, tab_lfs = st.tabs(["JSON", "Live for Speed Layout"])

    with tab_json:
        st.info(
            "The JSON object has 3 keys: `x`, `y` and `color`. `x` and `y` are lists of floats representing the x and y coordinates of the cones. `color` is a list of strings representing the color of the cones. The colors are either `blue`, `yellow` or `orange_big`. The length of the three lists should be the same. The cones appear in the same order as the track direction. The first cone is the start/finish line. The cones are ordered in the direction of the track."
        )
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
            help="This file should be placed inside the `data/layout` folder of your LFS installation.",
        )


if __name__ == "__main__":
    main()
