import streamlit as st

from utils.opencv.images import (
    bytes_to_image,
    read_image
    )

from utils.gui.images import (
    show_code,
    show_image,
    show_note
)


# This brings the whole rendition together
def render(img_file, img_file_name, upload=False):
    # checks if it's an upload
    if upload:
        show_code(img_file_name)
        show_image(bytes_to_image(img_file.read()))
        st.success("You are viewing results for your uploaded image")
        show_note(img_file_name)
        
    else:
        show_code(img_file_name)
        show_image(read_image("app/assets/Lenna.png"))
        st.error("Please upload an image to see different results")
        show_note(img_file_name)
        

    
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
        render(img_file, img_file_name, upload=True)
    else:
        render(img_file, img_file_name)
        


