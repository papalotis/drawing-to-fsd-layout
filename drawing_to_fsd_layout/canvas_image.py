import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from drawing_to_fsd_layout.image_processing import Image


def show_canvas_warning():
    st.warning("You need to draw a track in order to continue.")
    st.stop()


def get_canvas_image() -> Image:
    stroke_width = st.slider("Stroke width", 1, 25, 10)

    should_erase = st.radio("Eraser", ["Off", "On"], horizontal=True)

    stroke_color = "white" if should_erase == "On" else "black"

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is None:
        show_canvas_warning()

    # by default canvas changes the alpha channel, not the rgb channels
    raw_image_data = canvas_result.image_data
    image_data = canvas_result.image_data[:, :, [-1]]
    image_lightness = np.max(raw_image_data[:, :, :-1], axis=2) > 0

    image_data[image_lightness] = 0

    image_data = np.broadcast_to(image_data, image_data.shape[:-1] + (3,))

    if np.all(image_data == 255):
        show_canvas_warning()

    return image_data
