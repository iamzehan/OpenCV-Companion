import streamlit as st

st.set_page_config(page_icon="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png")
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
win.link_button("🪟 Windows", "https://docs.opencv.org/3.4/d5/de5/tutorial_py_setup_in_windows.html", type="primary", use_container_width=True)
fed.link_button("🎩 Fedora", "https://docs.opencv.org/3.4/dd/dd5/tutorial_py_setup_in_fedora.html", type="primary", use_container_width=True)
ubu.link_button("![image](https://upload.wikimedia.org/wikipedia/commons/b/b5/Former_Ubuntu_logo.svg) Ubuntu", "https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html", type="primary", use_container_width=True)

st.markdown("""<h2 align="center">Installing as a pip package</h2>""", unsafe_allow_html=True)
st.markdown("Run the following command:")
st.code("""pip install opencv-python""")
st.markdown("Check version:")
st.code("""
        import cv2 as cv
        print(cv.__version__)""", language="python", line_numbers=True)