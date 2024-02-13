import streamlit as st

from utils.opencv.images import (
    bytes_to_image,
    read_image
    )

# This brings the whole rendition together
def Read_and_Show_Image(img_file, img_file_name, render, upload=False):        
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
    
    # checks if it's an upload
    if upload:
        with render:
            show_code(img_file_name)
            st.subheader("Output")
            show_image(bytes_to_image(img_file.read()))
            st.success("You are viewing results for your uploaded image")
            st.subheader("Note")
            show_note(img_file_name)
        
    else:
        with render:
            show_code(img_file_name)
            st.subheader("Output")
            show_image(read_image("app/assets/Images/Lenna.png"))
            st.error("Please upload an image to see different results")
            st.subheader("Note")
            show_note(img_file_name)