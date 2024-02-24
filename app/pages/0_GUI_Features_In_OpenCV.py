import streamlit as st 
from utils.gui.footer import footer
from utils.gui.menu import menu

st.title("GUI Features in OpenCV")

with st.container(border=True):
    st.page_link("pages/1_Getting_Started_with_Images.py", label="👉 Getting Started with Images", use_container_width=True)
    col1, col2, col3 = st.columns([2,8,2])
    col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/image_display.jpg")
    col2.markdown("Learn to load an image, display it and save it back")
    col3.page_link("pages/1_Getting_Started_with_Images.py", label="Learn ➡️", use_container_width=True)

with st.container(border=True):
    st.page_link("pages/2_Getting_Started_with_Videos.py", label="👉 Getting Started with Videos 📽️")
    col1, col2, col3 = st.columns([2,8,2])
    col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/video_display.jpg")
    col2.markdown("Learn to play videos, capture videos from Camera and write it as a video")
    col3.page_link("pages/2_Getting_Started_with_Videos.py", label="Learn ➡️")

with st.container(border=True):
    st.page_link("pages/3_Drawing_Functions_in_OpenCV.py", label=" 👉Drawing Functions in OpenCV 🖌️")
    col1, col2, col3 = st.columns([2,8,2])
    col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/drawing1.jpg")
    col2.markdown("Learn to draw lines, rectangles, ellipses, circles etc with OpenCV")
    col3.page_link("pages/3_Drawing_Functions_in_OpenCV.py", label="Learn ➡️")
    
with st.container(border=True):
    st.page_link("pages/4_Mouse_Events.py", label="👉 Mouse as a Paint-Brush", use_container_width=True)
    col1, col2, col3 = st.columns([2,8,2])
    col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/mouse_drawing.jpg")
    col2.markdown("Draw stuff with your mouse")
    col3.page_link("pages/4_Mouse_Events.py", label="Learn ➡️")
    
with st.container(border=True):
    st.page_link("pages/5_Trackbar.py", label="👉 Trackbar as the Color Palette", use_container_width=True)
    col1, col2, col3 = st.columns([2,8,2])
    col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/trackbar.jpg")
    col2.markdown("Create trackbar to control certain parameters")
    col3.page_link("pages/5_Trackbar.py", label="Learn ➡️")

col1, col2, col3 = st.columns([2,8,2])
col3.page_link("pages/1_Getting_Started_with_Images.py", label="Next ➡️", use_container_width=True)
menu()
footer()