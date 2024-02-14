import streamlit as st
from PIL import ImageColor
from utils.opencv.images import (
    bytes_to_image,
    read_image,
    load_by_pixels,
    get_shape,
    list_to_np_array
    )

# Getting Started with Images (Page - 2)

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
        `"s"-key`. For this the `cv::imwrite` function is called that has the file
        path and the `cv::Mat` object as an argument.
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

# Basic Operations On Images (Page - 6)

def Accessing_Modifying_Pixel_Values(img_file, img_file_name, render, upload=False):
    
    def show_code(img_file_name):
        return f"""
            import numpy as np
            import cv2 as cv
            img = cv.imread('<path>/{img_file_name}')
            assert img is not None, "file could not be read, check with os.path.exists()"
            cv.imshow('image', img)
            """
    
    def show_image(img_file):
        with st.container(border=True):
            st.image(img_file, caption="image", channels='BGR', use_column_width=True)
    
    
    
    if upload:
        img = bytes_to_image(img_file.read())
        with render:
            st.code(show_code(img_file_name))
            st.subheader("Output")
            show_image(img)
        
    else:
        img = read_image(f"app/assets/Images/{img_file_name}")
        with render:
            st.code(show_code(img_file_name))
            st.subheader("Output")
            show_image(img)
    
    with render:
        
        st.write("""
                 You can access a pixel value by its row and column coordinates. 
                 For BGR image, it returns an array of Blue, Green, Red values. 
                 For grayscale image, just corresponding intensity is returned.""")
        
        row_max, column_max, channels = get_shape(img)
        
        with st.container(border=True):
            st.subheader("Playground")
            dimensions = st.slider("Row (Height): ", value=100, max_value=row_max),\
                        st.slider("Column (Width):", value=100, max_value=column_max)
        st.code(f"""
                >>> px = img[{dimensions[0]},{dimensions[1]}]
                >>> print( px )
                """)
        with st.expander("Output: "):
            st.markdown(load_by_pixels(img, dimensions)[0])
        
        color = st.selectbox("Get pixel by specific color?:",options=["Blue", "Green", "Red"])
        colors = {'Blue':0, 'Green': 1, 'Red':2}
        
        st.code(f"""
                # accessing only {color} pixel
                blue = img[{dimensions[0]},{dimensions[1]}, {colors[color]}]
                print( blue )
                """)
        color_result = load_by_pixels(img, dimensions, colors[color])
        with st.expander("Output: "):
            st.markdown(color_result)
        
        st.markdown("""
                    You can modify the pixel values the same way.
                    """)
        
        select_color=st.color_picker("Select color", value="#fff")
        select_color=list(ImageColor.getcolor(f'{select_color}','RGB')[::-1])
        
        st.code(f"""
                img[{dimensions[0]},{dimensions[1]}] = {select_color}
                print( img[{dimensions[0]},{dimensions[1]}] )
                """)
        
        with st.expander("Output: "):
            st.markdown(list_to_np_array(select_color))
        
        st.markdown(f"""
                    **Warning**\n
                    Numpy is an optimized library for fast array calculations. 
                    So simply accessing each and every pixel value and modifying 
                    it will be very slow and it is discouraged.
                    
                    **Note**
                    > The above method is normally used for selecting a region of
                    an array, say the first 5 rows and last 3 columns. 
                    For individual pixel access, the Numpy array methods,
                    array.item() and array.itemset() are considered better.
                    They always return a scalar, however, so if you want to 
                    access all the B,G,R values, you will need to call array.item() 
                    separately for each value.\n
                    Better pixel accessing and editing method :
                    """)
        
        modify_value=st.slider("Modify Value:", value = 100, max_value=255)
        
        st.code(f"""
                # accessing {color.upper()} value
                img.item({dimensions[0]}, {dimensions[1]}, {colors[color]})
                >> {color_result}
                # modifying {color.upper()} value
                img.itemset({dimensions[0]}, {dimensions[1]}, {modify_value})
                img.item({dimensions[0]}, {dimensions[1]}, {colors[color]})
                >> {modify_value}
                """)

def Accessing_Image_Properties():
    pass

def Image_ROI():
    pass 

def Splitting_and_Merging_Image_Channels():
    pass

def Making_Borders_for_Images():
    pass
