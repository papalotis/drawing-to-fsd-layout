from drawing_to_fsd_layout.image_processing import Image
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np

def show_canvas_warning():
    st.warning("You need to draw a track in order to continue.")
    st.stop()

def get_canvas_image() -> Image:
    
    stroke_width = st.slider("Stroke width", 1, 25, 10)

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        drawing_mode='freedraw',
        key="canvas",
    )

    if canvas_result.image_data is None:
        show_canvas_warning()

    # by default canvas changes the alpha channel, not the rgb channels
    image_data = canvas_result.image_data[:, :, [-1]]
    image_data = np.broadcast_to(image_data, image_data.shape[:-1] + (3,))
    image_data = 255 - image_data

    if np.all(image_data == 255):
        show_canvas_warning()
    
    return image_data
