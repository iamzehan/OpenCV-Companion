import streamlit as st
from utils.gui.footer import footer

st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png")
st.markdown("""
            <h1 align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png" height="40" width="auto"> Companion</h1> 
            <h6 align="center"> A Companion website to play around with OpenCV in realtime</h6>
            As a beginner I used to struggle a lot understanding most of the concepts in the OpenCV spectrum,
            it was hard to visualize the solutions without hardcoding them first. 
            So I decided to create a solution that allows you to do just that. It's no different than the original documentation.
            The only advantage it gives you is that, you get to modify the code with a GUI and understand what it does.
            <br> - <i>Md. Ziaul Karim</i>
            <h2 align="center"> Installation Guide </h2>
            """,
            unsafe_allow_html=True)
win, fed, ubu = st.columns([4,4,4])

st.markdown("""<style>
    .button-link {
        display: inline-block;
        padding: 10px 40px;
        background-color: #4BFF90;
        color: #FFFFFF!important;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
</style>
<table align="center">
<tr style="border: none;">
<td style="border: none;">
<a href="https://docs.opencv.org/3.4/d5/de5/tutorial_py_setup_in_windows.html" class="button-link" target="_blank">
    🪟 Windows
</a>
</td>
<td style="border: none;">
<a href="https://docs.opencv.org/3.4/dd/dd5/tutorial_py_setup_in_fedora.html" class="button-link" target="_blank"> 
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Fedora_icon_%282021%29.svg" alt="Ubuntu" style="vertical-align:middle;height:20px;"> 
Fedora
</a>
</td>
<td style="border: none;">
<a href="https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html" class="button-link" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Former_Ubuntu_logo.svg" alt="Ubuntu" style="vertical-align:middle;height:20px;"> Ubuntu
</a>
</td>
</tr>
</table>
""", unsafe_allow_html=True)
st.markdown("""<h2 align="center">Installing as a pip package</h2>""", unsafe_allow_html=True)
st.markdown("Run the following command:")
st.code("""pip install opencv-python""")
st.markdown("Check version:")
st.code("""
        import cv2 as cv
        print(cv.__version__)""", language="python", line_numbers=True)
footer()
