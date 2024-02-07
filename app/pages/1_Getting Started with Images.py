import streamlit as st
from utils.images import (
    bytes_to_image,
    read_image
    )

# 1. This shows the code
def show_code(img_file_name):
    return st.code(f"""
            import cv2 as cv
            import sys
            img = cv.imread("<path>/{img_file_name}")
            if img is None:
                sys.exit("Could not read the image.")
            cv.imshow("Display window", img) 
            k = cv.waitKey(0)
            if k == ord('s'):
                cv.imwrite("<path>/{img_file_name}", img)       
            """, line_numbers=True)
# 2. This renders the image
def show_image(img_file):
    with st.container(border=True):
        _, col, _ = st.columns([4,4,4])
        col.markdown("<center>Display window</center>", 
                        unsafe_allow_html=True)
        col.image(img_file)
        st.markdown("""Yes the color looks weird, 
                    because OpenCV reads image in BGR format. 
                    We'll learn about those in the future.""")
# 3. This shows the footnote
def show_note(img_file_name):
    st.markdown("""Because we want our window to be displayed
                until the user presses a key (otherwise the program
                would end far too quickly), we use the `cv::waitKey` 
                function whose only parameter is just how long should
                it wait for a user input (measured in milliseconds). 
                Zero means to wait forever. 
                The return value is the key that was pressed.""")
    
    st.code("""
        cv.imshow("Display window", img)
        k = cv.waitKey(0)""")
    st.markdown("""
    In the end, the image is written to a file if the pressed key was the 
    "s"-key. For this the cv::imwrite function is called that has the file
    path and the cv::Mat object as an argument.
""")
    st.code(f"""
if k == ord("s"):
cv.imwrite("<path>/{img_file_name}", img)""")
    
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
        


