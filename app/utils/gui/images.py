import streamlit as st
from PIL import ImageColor
from utils.opencv.images import (
    bytes_to_image,
    read_image,
    load_by_pixels,
    get_shape,
    get_size,
    get_dtype,
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
class BasicOperations:
    def __init__(self):
        self.img = None 
        self.img_file=None
    
    def show_code(self, img_file_name):
        st.subheader("Code")
        st.code(f"""
            import numpy as np
            import cv2 as cv
            img = cv.imread('<path>/{img_file_name}')
            assert img is not None, "file could not be read, check with os.path.exists()"
            cv.imshow('image', img)
            """)
        
    def show_image(self, img_file):
        st.subheader("Output")
        with st.container(border=True):
            _, col, _ = st.columns([4,4,4])
            col.image(img_file, channels='RGB', caption='image')
    
    def main_body(self, show=True):
        
        img_file_name = 'messi5.jpg'
        img_file = self.img_file
        
        with st.expander("Main Code", expanded=show): 
            if img_file:
                self.img = bytes_to_image(img_file.read())
                self.show_code(img_file.name)
                st.success("Results from your uploaded image")
                self.show_image(self.img)
                
                
            else:
                self.img = read_image(f"app/assets/Images/{img_file_name}")
                self.show_code(img_file_name)
                st.info("Results from the example image")
                self.show_image(self.img)
                st.sidebar.error("Upload an image to see changes")
                
            
    def side_bar(self):
        # File and name handling
        self.img_file = st.sidebar.file_uploader("Upload an Image to see how the code changes:", type=["PNG","JPG"], label_visibility="collapsed")
        
    def Accessing_Modifying_Pixel_Values(self):
        st.markdown("""
                    ## Accessing and Modifying pixel values
                    Let's load a color image first:
                """)
        self.main_body()
        img = self.img
        with st.container():
            
            st.write("""
                    You can access a pixel value by its row and column coordinates. 
                    For BGR image, it returns an array of Blue, Green, Red values. 
                    For grayscale image, just corresponding intensity is returned.
                    """)
            
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

    def Accessing_Image_Properties(self):
        st.markdown("""
                    ## Accessing Image Properties
                    Continuation from our previous task:
                    """)
        self.main_body(show=False)
        img = self.img
        st.markdown("""
                    Image properties include number of rows, columns, and channels;
                    type of image data; number of pixels; etc.
                    The shape of an image is accessed by img.shape.
                    It returns a tuple of the number of rows, columns, and channels 
                    (if the image is color):
                    """)
        st.code(f"""
                print( img.shape )
                >> {get_shape(img)}
                """)
        st.markdown("""
                    Note
                    > If an image is grayscale, the tuple returned contains only the
                    number of rows and columns, so it is a good method to check whether
                    the loaded image is grayscale or color.
                    Total number of pixels is accessed by `img.size`:
                    """)
        st.code(f"""
                    print( img.size )
                    >>> {get_size(img)}
                    """)
        st.markdown(f"""
                    Image datatype is obtained by `img.dtype`:
                    ```python
                    print( img.dtype )
                    >>> {get_dtype(img)}
                    ```
                    """)
        st.markdown(f"""
                    **Note**
                    > `img.dtype` is very important while debugging because a large
                    number of errors in OpenCV-Python code are caused by invalid 
                    datatype.
                    """)

    def Image_ROI(self):
        st.markdown("""
                    ## Image ROI
                    Continuation from our previous task:
                    """)
        self.main_body(show=False)
        st.markdown("""
                    Sometimes, you will have to play with certain regions of images.
                    For eye detection in images, first face detection is done over 
                    the entire image. When a face is obtained, we select the face 
                    region alone and search for eyes inside it instead of searching 
                    the whole image. It improves accuracy (because eyes are always on 
                    faces :D ) and performance (because we search in a small area).
                    ROI is again obtained using Numpy indexing. Here I am selecting 
                    the ball and copying it to another region in the image:
                    """)
        img = self.img
        
        row_max, column_max, channels = get_shape(img)
            
        with st.container(border=True):
            st.subheader("Playground")
            st.markdown("<center>Row range</center>", unsafe_allow_html=True)
            y_0, y_1 = st.slider("$y_0$", value=344, max_value=row_max-1),\
                        st.slider("$y_1$", value = 404, max_value = row_max)
            st.markdown("<center>Column range</center>", unsafe_allow_html=True)
            x_0, x_1 = st.slider("$x_0$", value=379, max_value=column_max-1),\
                        st.slider("$x_1$", value = 447, max_value = column_max)
            y_diff, x_diff = abs(y_0-y_1), abs(x_0-x_1)
            st.markdown("<center>Relocate to</center>", unsafe_allow_html=True)
            locate = st.slider("$y$", value=342, max_value=row_max-y_diff),\
                    st.slider("$x$", value = 190, max_value = column_max-x_diff)
        try:              
            st.code(f"""
                    >>> ball = img[{y_0}:{y_1}, {x_0}:{x_1}]
                    >>> img[{locate[0]}:{locate[0]+y_diff}, {locate[1]}: {locate[1] + x_diff}] = ball
                    """)
            ball = img[y_0:y_1, x_0:x_1]
            img[locate[0]:locate[0]+y_diff, locate[1]:locate[1]+x_diff] = ball
            st.image(ball)
            st.image(img)
        except:
            st.error("An Error has occured")
        
    def Splitting_and_Merging_Image_Channels(self):
        st.markdown("""
                    ## Splitting and Merging Image Channels
                    """)
        self.main_body(show=False)
        st.markdown("""
                    Sometimes you will need to work separately on the B,G,R channels
                    of an image. In this case, you need to split the BGR image into 
                    single channels. In other cases, you may need to join these 
                    individual channels to create a BGR image. 
                    You can do this simply by:
                    """)

    def Making_Borders_for_Images(self):
        self.main_body()
