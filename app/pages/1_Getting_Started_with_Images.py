import streamlit as st
from utils.gui import footer, menu, images

if __name__ == "__main__":
    
    # Page Configurations
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                   page_title="Getting Started with Images")
    
    menu.menu()
    # The Goals of the lesson
    st.markdown("""
                # Getting Started with Images 🖼️
                ## Goals
                * Learning to read images from file with `imread()`
                * Learning to Display an image in an OpenCV window with `imshow()`
                * Learning to Write an Image to a file with `imwrite()`
                """,
                unsafe_allow_html=True)
    
    guiFeat = images.GUIFeatures()
    guiFeat.Read_and_Show_Image()
    
    footer.footer()