import streamlit as st 
from utils.gui.videos import (
    Capture_Video_from_Webcam,
    Play_Video_from_File,
    Save_Video
)
from utils.gui.menu import menu
from utils.gui.footer import footer

if __name__ == "__main__":
    
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Getting Started with Videos")

    menu()
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select:", options=["Introduction",
                                                        "Capture Video from Camera",
                                                            "Playing Video from File",
                                                            "Save Video"],
                            horizontal=True,
                            label_visibility="collapsed" )
    
    if options == "Introduction":
        st.markdown("""
                        # Getting Started with Videos 📽️
                        ## Goals
                        * Learn to read video, display video, and save video.
                        * Learn to capture video from a camera and display it.
                        * You will learn these functions : `cv.VideoCapture()`, `cv.VideoWriter()`
                    """,
                unsafe_allow_html=True)
        
    if options == "Capture Video from Camera":
        Capture_Video_from_Webcam()
        
    elif options == "Playing Video from File":
        Play_Video_from_File()
        
    elif options== "Save Video":
        Save_Video()
    footer()