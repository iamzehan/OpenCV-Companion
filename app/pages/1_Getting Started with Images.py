import streamlit as st
from utils.gui.images import Read_and_Show_Image

if __name__ == "__main__":
    
    # Page Configurations
    st.set_page_config(page_icon="app\\assets\\OpenCV_Logo_with_text.png",
                   page_title="Getting Started with Images")
    
    # The Goals of the lesson
    st.markdown("""
                # Getting Started with Images 🖼️
                ## Goals
                * Learning to read images from file with `imread()`
                * Learning to Display an image in an OpenCV window with `imshow()`
                * Learning to Write an Image to a file with `imwrite()`
                """,
                unsafe_allow_html=True)
    
    # File and name handling
    img_file_name = 'Lenna.png'
    img_file = st.file_uploader("Upload an Image to see how the code changes:", type=["PNG","JPG"])
    
    # Checks if a File has been uploaded
    if img_file:
        # extracting name img_file object of the Upload class
        img_file_name = img_file.name
        # rendition of the whole view
        Read_and_Show_Image(img_file, img_file_name, upload=True)
    else:
        Read_and_Show_Image(img_file, img_file_name)
        


