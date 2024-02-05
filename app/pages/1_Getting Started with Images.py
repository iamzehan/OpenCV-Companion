import streamlit as st
from utils.images import bytes_to_image

st.set_page_config(page_icon="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png", page_title="Getting Started with Images")
st.markdown("""
            # Getting Started with Images 🖼️
            ## Goals
            * Learning to read images from file with `imread()`
            * Learning to Display an image in an OpenCV window with `imshow()`
            * Learning to Write an Image to a file with `imwrite()`
            """,
            unsafe_allow_html=True)


img = None

with st.expander("Sample code"):
    st.code("""
                import cv2 as cv
                import sys
                img = cv.imread("<path>/my_image.jpg"))
                if img is None:
                    sys.exit("Could not read the image.")
                cv.imshow("Display window", img)
                k = cv.waitKey(0)
                if k == ord('s'):
                    cv.imwrite("<path>/my_image.jpg", img)
                """, line_numbers=True)
    
if not img:
    img = st.file_uploader("Upload an Image to see how the code changes:")
    if img:
        st.code(f"""
                import cv2 as cv
                import sys
                img = cv.imread("{img.name}")
                if img is None:
                    sys.exit("Could not read the image.")
                cv.imshow("Display window", img) 
                k = cv.waitKey(0)
                if k == ord('s'):
                    cv.imwrite("<path>/{img.name}", img)       
                """)
        with st.expander("`cv.imshow('Display window', img)`",expanded=True):
            _, col, _ = st.columns([4,4,4])
            col.markdown("Display window")
            col.image(bytes_to_image(img.read()))
            st.markdown("""Yes the color looks weird, 
                        because OpenCV reads image in BGR format. 
                        We'll learn about those in the future.""")
            
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
    cv.imwrite("<path>/{img.name}", img)""")
    else:
        st.error("Please upload an image")


        


