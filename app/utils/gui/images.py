import streamlit as st
from PIL import ImageColor
from utils.opencv.images import (
    bytes_to_image,
    read_image,
    load_by_pixels,
    get_shape,
    get_size,
    get_dtype,
    list_to_np_array,
    split_channels,
    merge_channels,
    make_borders,
    add_two_img
    )

# Getting Started with Images (Page - 2)

class CommonComponents:
    def __init__(self):
        pass
    
    def side_bar(self, multiple=False):
        
        # File and name handling
        file = st.sidebar.file_uploader("Upload an Image:",
                                        type=["PNG","JPG"], 
                                        label_visibility="collapsed",
                                        accept_multiple_files=multiple)
        message = st.sidebar.empty()
        if not file:
            message.error("Upload an image to see changes")
            
        else:
            if multiple:
                try:
                    file1, file2, *_ = file
                    self.img_file1, self.img_file2 = file1, file2
                    self.img_file_name1, self.img_file_name2 = file1.name, file2.name
                    self.img1, self.img2 = bytes_to_image(file1.read()), bytes_to_image(file2.read())
                except:
                    message.error("You must upload two images")
                
            else:
                self.img_file = file
                self.img_file_name = file.name
                self.img = bytes_to_image(file.read())

class GUIFeatures(CommonComponents):
    
    def __init__(self):
        self.img = read_image("app/assets/Images/Lenna.png") 
        self.img_file=None
        self.img_file_name = 'Lenna.png'
        
    # 1. This shows the code
    def show_code(self, img_file_name):
        st.subheader("Code")
        st.code(f"""
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
    def show_image(self, img_file):
        st.subheader("Output")
        with st.container(border=True):
            _, col, _ = st.columns([4,4,4])
            col.markdown("<center>Display window</center>", 
                            unsafe_allow_html=True)
            col.image(img_file, channels="BGR")
            
    # 3. This shows the footnote
    def show_note(self, img_file_name):
        st.subheader("Note")
        st.markdown("""
                    Because we want our window to be displayed
                    until the user presses a key (otherwise the program
                    would end far too quickly), we use the `cv::waitKey` 
                    function whose only parameter is just how long should
                    it wait for a user input (measured in milliseconds). 
                    Zero means to wait forever. 
                    The return value is the key that was pressed.
                """)
        
        st.code("""
                    cv.imshow("Display window", img)
                    k = cv.waitKey(0)
                """)
        
        st.markdown("""
                    In the end, the image is written to a file if the pressed key was the 
                    `"s"-key`. For this the `cv::imwrite` function is called that has the file
                    path and the `cv::Mat` object as an argument.
                """)
        
        st.code(f"""
                    if k == ord("s"):
                    cv.imwrite("<path>/{img_file_name}", img)""")
    
    def main_body(self):
        
        img_file_name = self.img_file_name
        img_file = self.img_file
        img = self.img
        
        with st.container(border=True): 
            if img_file:
                self.show_code(img_file_name)
                self.show_image(img)
                st.success("Results from your uploaded image")
                self.show_note(img_file_name)
                
            else:
                self.show_code(img_file_name)
                self.show_image(img)
                st.info("Results from the example image")
                self.show_note(img_file_name)
                
    # This brings the whole rendition together
    def Read_and_Show_Image(self):
        self.side_bar()        
        self.main_body()

# Basic Operations On Images (Page - 6)
class BasicOperations(CommonComponents):
    
    def __init__(self):
        self.img = read_image(f"app/assets/Images/messi5.jpg")
        self.img_file=None
        self.img_file_name = 'messi5.jpg'
    
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
            col.image(img_file, channels="BGR", caption='image')
    
    def main_body(self, show=True):
        
        img_file_name = self.img_file_name
        img_file = self.img_file
        img = self.img
        
        with st.expander("Main Code", expanded=show): 
            if img_file:
                self.show_code(img_file_name)
                self.show_image(img)
                st.success("Results from your uploaded image")
                
            else:
                self.show_code(img_file_name)
                self.show_image(img)
                st.info("Results from the example image")
                
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
        
        st.subheader("Image ROI")
        st.markdown("Continuation from our previous task:")
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
        default = [344, 404, 379, 447]
        
        with st.container(border=True):
            st.subheader("Playground")
            
            y_0, y_1 = st.slider("Row range", value=(344, 404), min_value=1, max_value=row_max)
            x_0, x_1 = st.slider("Column range", value=(379, 447), min_value=1, max_value=column_max)
            
            y_diff, x_diff = abs(y_0 - y_1), abs(x_0 - x_1)
            locate_0, locate_1 = row_max-y_diff, column_max - x_diff
            locate = st.slider("$y$ `Up-Down`", value=locate_0, max_value = locate_0),\
                    st.slider("$x$ `Left-Right`", value = locate_1, max_value = locate_1)
            
            if [y_0, y_1, x_0, x_1] != default: 
                st.code(f"""
                            >>> ball = img[{y_0}:{y_1}, {x_0}:{x_1}]
                            >>> img[{locate[0]}:{locate[0]+y_diff}, {locate[1]}: {locate[1]+x_diff}] = ball
                        """)
                ball = img[y_0:y_1, x_0:x_1]
                
            else:
                st.code(f"""
                            >>> ball = img[344:404, 379:447]
                            >>> img[{locate[0]}:{locate[0]+y_diff}, {locate[1]}: {locate[1] + x_diff}] = ball
                        """)
                y_0, y_1, x_0, x_1 = default
                ball = img[y_0:y_1, x_0:x_1]
                         
            img[locate[0]:locate[0]+y_diff, locate[1]:locate[1]+x_diff] = ball
            
            with st.container(border=True):
                st.markdown("<center> Output </center>", unsafe_allow_html=True)
                col1, col2 = st.columns([2, 8])
                col1.image(ball, caption='ball', use_column_width=True, channels='BGR')
                col2.image(img, caption = 'Original image with replaced pixel', channels='BGR' )  

    def Splitting_and_Merging_Image_Channels(self):
        st.markdown("""
                    ## Splitting and Merging Image Channels
                    """)
        self.main_body(show=False)
        img = self.img
        st.markdown("""
                    Sometimes you will need to work separately on the B,G,R channels
                    of an image. In this case, you need to split the BGR image into 
                    single channels. In other cases, you may need to join these 
                    individual channels to create a BGR image. 
                    You can do this simply by:
                    """)
        st.code("""
                b,g,r = cv2.split(img)
                img = cv2.merge((b,g,r))
                """)
        with st.expander('Output', expanded=False):
            
            col1, col2, col3 = st.columns([4,4,4])
            
            b, g, r = split_channels(img)
            img = merge_channels(b, g, r)
            
            col1.image(b, 'b')
            col2.image(g, 'g')
            col3.image(r, 'r')
            st.image(img, 'Merged', channels='BGR', use_column_width=True)
        
        st.markdown("""
                    Or:
                    
                    ```python
                    b = img[:,:,0]
                    g = img[:,:,1]
                    r = img[:,:,2]
                    ```
                    Suppose, you want to make all the blue pixels to zero,
                    you need not split like this and put it equal to zero.
                    You can simply use Numpy indexing which is faster.
                    """)
        color_ch = st.radio("Select which channel you want to turn to zero: ",
                            options=['b', 'g', 'r'], horizontal=True)
        colors = {'b' : 0, 'g' : 1, 'r' : 2} 
        st.code(f"""
                img[:, :, {colors[color_ch]}]=0
                """)
        with st.expander('Output', expanded=False):
            img[:, :, colors[color_ch]]=0
            st.image(img, f"With Zero '{color_ch}'", use_column_width=True)
        
        st.warning("""
                   **⚠️Warning**
                   > `cv2.split()` is a costly operation (in terms of time),
                   so only use it if necessary. Numpy indexing is much more 
                   efficient and should be used if possible.
                   """)
    
    def Making_Borders_for_Images(self):
        
        def show_code(img_file_name):
            st.code(f"""
                import cv2
                import numpy as np
                from matplotlib import pyplot as plt

                BLUE = [255,0,0]

                img1 = cv2.imread('{img_file_name}')

                replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
                reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
                reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
                wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
                constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

                plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
                plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
                plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
                plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
                plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
                plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

                plt.show()
                """)
            
        def show_image(img):
            info = st.empty()
            button_space = st.empty()
            container_space = st.empty()
            if button_space.button("▶️"):
                replicate, reflect, reflect101, wrap, constant = make_borders(img)
                with container_space.container(border=True):
                    col1, col2, col3 = st.columns([4,4,4])
                    
                    col1.markdown("<center>ORIGINAL</center>", 
                                    unsafe_allow_html=True)
                    col1.image(img, channels='BGR')
                    
                    col2.markdown("<center>REPLICATE</center>", 
                                    unsafe_allow_html=True)
                    col2.image(replicate, channels='BGR')
                    
                    col3.markdown("<center>REFLECT</center>", 
                                    unsafe_allow_html=True)
                    col3.image(reflect, channels='BGR')
                    
                    col1.markdown("<center>REFLECT 101</center>", 
                                    unsafe_allow_html=True)
                    col1.image(reflect101, channels='BGR')
                    
                    col2.markdown("<center>WRAP</center>", 
                                    unsafe_allow_html=True)
                    col2.image(wrap, channels='BGR')
                    
                    col3.markdown("<center>CONSTANT</center>", 
                                    unsafe_allow_html=True)
                    col3.image(constant, channels='BGR')
                    st.success("Showing Results")
                    info.error("Press ❌ to exit")
                    
                if button_space.button("❌"):
                    return
            else:
                info.info("Push ▶️ to see output")
 
        st.markdown(f"""
                    # Making Borders for Images (Padding)
                    """)
        
        
        st.markdown("""
                    If you want to create a border around the image, 
                    something like a photo frame, you can use `cv2.copyMakeBorder()` 
                    function. 
                    But it has more applications for convolution operation,
                    zero padding, etc. This function takes the following arguments:

                    - **src**: input image
                    - **top, bottom, left, right**: border width in the number of 
                    pixels in corresponding directions
                    - **borderType**: Flag defining what kind of border to be added.
                    It can be the following types:
                    - `cv2.BORDER_CONSTANT`: Adds a constant colored border. 
                    The value should be given as the next argument.
                    - `cv2.BORDER_REFLECT`: Border will be mirror reflection of the
                    border elements, like this: fedcba|abcdefgh|hgfedcb
                    - `cv2.BORDER_REFLECT_101` or `cv2.BORDER_DEFAULT`: Same as above,
                    but with a slight change, like this: gfedcb|abcdefgh|gfedcba
                    - `cv2.BORDER_REPLICATE`: Last element is replicated throughout,
                    like this: aaaaaa|abcdefgh|hhhhhhh
                    - `cv2.BORDER_WRAP`: Can’t explain, it will look like this:
                    cdefgh|abcdefgh|abcdefg
                    - **value**: Color of the border if border type is 
                    `cv2.BORDER_CONSTANT`

                    Below is a sample code demonstrating all these border types for 
                    better understanding:
                    """)
        img_file = self.img_file
        img_file_name = self.img_file_name
        img = self.img
        if img_file:
            show_code(img_file_name)
            show_image(img)
        else:
            show_code(img_file_name)
            show_image(img)

class ArithmeticOperations(CommonComponents):
    def __init__(self):
        self.img_file1, self.img_file2 = None, None
        self.img_file_name1, self.img_file_name2 = 'ml.png', \
                                                    'OpenCV_Logo_with_text.png'
        self.img1, self.img2 = read_image('app/assets/Images/ml.png'),\
                                read_image('app/assets/Images/OpenCV_Logo_with_text.png')

    def Image_Addition(self):
        st.markdown("""
                    ## Image Addition
                    You can add two images by OpenCV function, 
                    `cv2.add()` or simply by numpy operation, 
                    `res = img1 + img2`. Both images should be 
                    of same depth and type, or second image 
                    can just be a scalar value.
                    """)
        st.info("""
                ⚠️ Note
                > There is a difference between OpenCV addition 
                and Numpy addition. OpenCV addition is a saturated 
                operation while Numpy addition is a modulo operation.
                """)    
        
        st.markdown("""
                    For example, consider below sample:
                    ```python
                    >>> x = np.uint8([250])
                    >>> y = np.uint8([10])
                    >>> print cv2.add(x,y) # 250+10 = 260 => 255
                    [[255]]

                    >>> print x+y          # 250+10 = 260 % 256 = 4
                    [4]
                    ```
                    """)
        
        with st.expander("Example:", expanded=False):
            col1, col2 = st.columns(2)
            
            col1.image(self.img1, 'img1', channels= 'BGR', use_column_width=True)
            col2.image(self.img2, 'img2', channels= 'BGR', use_column_width=True)
            
            st.image(add_two_img(self.img1, self.img2), 'Image Addition', channels= 'BGR', use_column_width=True)
            
            st.warning(f"""
                       ⚠️Warning!
                       > The output is based on some preprocessing. 
                       Generally, the two images won't be added if they differ
                       in size as the width and height has to be the same, also the
                       image channels should match as well. So, we have to make sure
                       the two images match in their channel depth (Either both has 
                       to be color image, or both has to be black and white)
                       We have used
                       the following code to show the above output:
                       ```python
                    import cv2 as cv
                    
                    # gives the image shape (height, width, channels)
                    def get_shape(img):
                            return img.shape
                            
                    # performs resize on the image - as in stretch or squeeze 
                    def resize(img, h, w):
                            return cv.resize(img, (h, w))
                    
                    # performs addition on the image by matching height, width
                    def add_two_img(img1, img2):
                            h1, w1, _ = get_shape(img1)
                            h2, w2, _ = get_shape(img2)
                            h, w = max(h1, h2), max(w1, w2)
                            img1, img2 = resize(img1, h, w), resize(img2, h, w)
                            return cv.add(img1,img2) 
                            
                    img1 = cv.imread("{self.img_file_name1}") # your own img1 path
                    img2 = cv.imread("{self.img_file_name2}") # your own img2 path
                    
                    res = add_two_img(img1, img2)
                    cv.imshow('Image Addition', res)
                    cv.waitKey(0)
                       ```
                    Feel free to copy and run the code.
                       """)