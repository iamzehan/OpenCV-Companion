import streamlit as st
from PIL import ImageColor
from utils.opencv.images import (
    blank_image,
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
    add_two_img,
    bitwise_ops,
    performance_measure,
    colorspace_flags,
    object_tracking,
    find_hsv_values,
    scaling,
    translation,
    rotation
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
            if multiple:
                message.error("Upload images to see changes")
            else:
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
            st.image(img, f"With Zero '{color_ch}'", channels='BGR', use_column_width=True)
        
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
            info.info("Push ▶️ to see output")
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
        self.img1, self.img2 = read_image(f'app/assets/Images/{self.img_file_name1}'),\
                                read_image(f'app/assets/Images/{self.img_file_name2}')

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
                        """)
            st.code(f"""
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
                    """)
            st.markdown("""
                    Feel free to copy and run the code.
                       """)
    
    def Image_Blending(self):
        
        def show_code(img_file_name1, img_file_name2):
            st.subheader("Code")
            st.code(f"""
                    img1 = cv2.imread('{img_file_name1}')
                    img2 = cv2.imread('{img_file_name2}')

                    dst = cv2.addWeighted(img1,{alpha},img2,{beta},{gamma})

                    cv2.imshow('dst',dst)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                """)
            
        def show_image(img1, img2, alpha, beta, gamma):
            st.subheader("Ouput")
            col1, col2 = st.columns(2)
            
            col1.image(self.img1, 'img1', channels= 'BGR', use_column_width=True)
            col2.image(self.img2, 'img2', channels= 'BGR', use_column_width=True)
            
            st.image(add_two_img(self.img1, self.img2, alpha, beta, gamma, blend=True),
                     'dst', channels= 'BGR', use_column_width=True)

        st.markdown("""
                    ## Image Blending
                    This is also image addition, but different weights 
                    are given to images so that it gives a feeling of 
                    blending or transparency. Images are added as per 
                    the equation below:
                    """)
        
        st.latex(r"""
                    g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)
        """)
        
        st.markdown("""

                    By varying $$\\alpha$$ from $$0 \\rightarrow 1$$, 
                    you can perform a cool transition between 
                    one image to another.

                    Here I took two images to blend them together. 
                    First image is given a weight of 0.7 and second
                    image is given 0.3. `cv2.addWeighted()` applies 
                    following equation on the image.
                    """)
        
        st.latex(r"""
                    dst = \alpha \cdot img1 + \beta \cdot img2 + \gamma
                """)

        st.markdown("""
                    Here $$\gamma$$ is taken as zero.
                    """)
        
        with st.container(border=True):
            st.subheader("Parameters")
            defaults = [self.img_file_name1, self.img_file_name2, 0.7, 0.3, 0]
            alpha = st.slider('$\\alpha$:', value = 0.7, min_value=0.0, max_value=1.0)
            beta = st.slider('$\\beta$:', value = 0.3, min_value=0.0, max_value=1.0)
            gamma = st.slider('$\\gamma$ :', value=0, min_value=0, max_value=1)
            if defaults != [self.img_file_name1, self.img_file_name2, alpha, beta, gamma]:
                show_image(self.img1, self.img2, alpha, beta, gamma)
                st.success("Your ouput")
                show_code(self.img_file_name1, self.img_file_name2)
                st.success("Your Code")
            else:
                show_image(self.img1, self.img2, alpha, beta, gamma)
                st.info("Example ouput")
                show_code(self.img_file_name1, self.img_file_name2)
                st.info("Example Code")
    
    def Bitwise_Operations(self):
        
        def show_image(img1, img2):
            st.subheader("Output")
            st.image(bitwise_ops(img1, img2), channels='BGR', use_column_width=True)
        
        def show_code(img_file_name1, img_file_name2):
            st.subheader("Code")
            st.code(f"""
                # Load two images
                img1 = cv2.imread('{img_file_name1}')
                img2 = cv2.imread('{img_file_name2}')

                # I want to put logo on top-left corner, So I create a ROI
                rows,cols,channels = img2.shape
                roi = img1[0:rows, 0:cols ]

                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)
                img1[0:rows, 0:cols ] = dst

                cv2.imshow('res',img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """)
            
        st.markdown("""
                    ## Bitwise Operations
                    This includes bitwise AND, OR, NOT and XOR operations. 
                    They will be highly useful while extracting any part 
                    of the image (as we will see in coming chapters), 
                    defining and working with non-rectangular ROI etc. 
                    Below we will see an example on how to change a particular 
                    region of an image.

                    I want to put OpenCV logo above an image. 
                    If I add two images, it will change color. 
                    If I blend it, I get an transparent effect. 
                    But I want it to be opaque. If it was a rectangular region,
                    I could use ROI as we did in last chapter. But OpenCV logo 
                    is a not a rectangular shape. So you can do it with bitwise 
                    operations as below:
                    
                    """)
        defaults = ['ml.png', 'OpenCV_Logo_with_text.png']
        
        with st.container(border=True):
            if defaults != [self.img_file_name1, self.img_file_name2]:
                show_image(self.img1, self.img2)
                st.success("Your Output")
                show_code(self.img_file_name1, self.img_file_name2)
                st.success("Your code")
            else:
                show_image(self.img1, self.img2)
                st.info("Example Ouput")
                show_code(self.img_file_name1, self.img_file_name2)
                st.info("Example Code")

class PerformanceMeasurement(CommonComponents):
    def __init__(self):
        self.img = read_image(f"app/assets/Images/messi5.jpg")
        self.img_file=None
        self.img_file_name = 'messi5.jpg'
        
    def Measuring_Performance(self):
        st.markdown("""
                    `cv2.getTickCount` function returns the number of clock-cycles after
                    a reference event (like the moment machine was switched ON) to the
                    moment this function is called. So if you call it before and after
                    the function execution, you get number of clock-cycles used to 
                    execute a function.
                    `cv2.getTickFrequency` function returns the frequency of clock-cycles,
                    or the number of clock-cycles per second. So to find the time of
                    execution in seconds, you can do following:
                    """)
        st.code("""
                e1 = cv2.getTickCount()
                # your code execution
                e2 = cv2.getTickCount()
                time = (e2 - e1)/ cv2.getTickFrequency()
                """)
        st.markdown("""
                    We will demonstrate with following example.
                    Following example apply median filtering with
                    a kernel of odd size ranging from 5 to 49. 
                    (Don’t worry about what will the result look like,
                    that is not our goal):""")
        
        st.code(f"""
                img1 = cv2.imread('{self.img_file_name}')
                e1 = cv2.getTickCount()
                for i in xrange(5,49,2):
                    img1 = cv2.medianBlur(img1,i)
                e2 = cv2.getTickCount()
                t = (e2 - e1)/cv2.getTickFrequency()
                print t
                """)
        message = st.empty()
        message.info("Press ▶️ to see output")
        button_space = st.empty()
        code_space = st.empty()
        success=st.empty()
        if button_space.button("▶️"):
            code_space.code(f"""
                    {performance_measure(self.img)}
                    """)
            message.error("Press ❌ to close")
            success.success("Showing result")
            if button_space.button("❌"):
                code_space.empty()
                success.empty()
    
    def Default_Optimization(self):
        st.markdown("""
                    Many of the OpenCV functions are optimized using SSE2,
                    AVX etc. It contains unoptimized code also. 
                    So if our system support these features, 
                    we should exploit them (almost all modern day processors 
                    support them). It is enabled by default while compiling.
                    So OpenCV runs the optimized code if it is enabled, else 
                    it runs the unoptimized code. You can use `cv2.useOptimized()`
                    to check if it is enabled/disabled and `cv2.setUseOptimized()` 
                    to enable/disable it. Let’s see a simple example.
                    """)
        
        st.code("""
                # check if optimization is enabled
                In [5]: cv2.useOptimized()
                Out[5]: True

                In [6]: %timeit res = cv2.medianBlur(img,49)
                10 loops, best of 3: 34.9 ms per loop

                # Disable it
                In [7]: cv2.setUseOptimized(False)

                In [8]: cv2.useOptimized()
                Out[8]: False

                In [9]: %timeit res = cv2.medianBlur(img,49)
                10 loops, best of 3: 64.1 ms per loop
                """)
        
        st.markdown("""
                    See, optimized median filtering is ~2x faster
                    than unoptimized version. If you check its source,
                    you can see median filtering is SIMD optimized. 
                    So you can use this to enable optimization at the 
                    top of your code (remember it is enabled by default).
                    """)
        
    def Measuring_Performance_IPython(self):
        st.markdown("""
                    Sometimes you may need to compare the performance of two similar
                    operations. IPython gives you a magic command %timeit to perform
                    this. It runs the code several times to get more accurate results.
                    Once again, they are suitable to measure single line codes.

                    For example, do you know which of the following addition operation
                    is better, `x = 5; y = x**2`, `x = 5; y = x*x`, `x = np.uint8([5]);
                    y = x*x` or `y = np.square(x)` ? We will find it with %timeit in 
                    IPython shell.
                    """)
        
        st.code("""
                In [10]: x = 5

                In [11]: %timeit y=x**2
                10000000 loops, best of 3: 73 ns per loop

                In [12]: %timeit y=x*x
                10000000 loops, best of 3: 58.3 ns per loop

                In [15]: z = np.uint8([5])

                In [17]: %timeit y=z*z
                1000000 loops, best of 3: 1.25 us per loop

                In [19]: %timeit y=np.square(z)
                1000000 loops, best of 3: 1.16 us per loop
                """)
        
        st.markdown("""
                    You can see that, `x = 5 ; y = x*x` is fastest and it is around
                    20x faster compared to Numpy. If you consider the array creation
                    also, it may reach upto 100x faster. Cool, right? 
                    (Numpy devs are working on this issue)
                    """)
        
        st.info("""
                ⚠️ **Note**
                > Python scalar operations are faster than Numpy scalar operations.
                So for operations including one or two elements, Python scalar is 
                better than Numpy arrays. 
                Numpy takes advantage when size of array is a little bit bigger.
                """)
        
        st.markdown("""
                    We will try one more example. This time, we will compare the 
                    performance of `cv2.countNonZero()` and `np.count_nonzero()` 
                    for same image.
                    """)
        st.code("""
                In [35]: %timeit z = cv2.countNonZero(img)
                100000 loops, best of 3: 15.8 us per loop

                In [36]: %timeit z = np.count_nonzero(img)
                1000 loops, best of 3: 370 us per loop
                """)
        
        st.markdown("""
                    See, OpenCV function is nearly 25x faster than Numpy function.
                    """)
        
        st.info("""
                ⚠️ **Note**
                > Normally, OpenCV functions are faster than Numpy functions. 
                So for same operation, OpenCV functions are preferred. 
                But, there can be exceptions, 
                especially when Numpy works with views instead of copies.
                """)
        
    def Ipython_Magic_Commands(self):
        st.markdown("""
                    There are several other magic commands to measure the performance,
                    profiling, line profiling, memory measurement etc. 
                    They all are well documented. So only links to those docs
                    are provided here. 
                    Interested readers are recommended to try them out.
                    """)
    
    def Performance_Optimization_Techniques(self):
        st.markdown("""
                    There are several techniques and coding methods to exploit maximum performance of Python and Numpy. 
                    Only relevant ones are noted here and links are given to important sources. 
                    The main thing to be noted here is that, first try to implement the algorithm in a simple manner. 
                    Once it is working, profile it, find the bottlenecks and optimize them.

                    1. Avoid using loops in Python as far as possible, especially double/triple loops etc. 
                    They are inherently slow.
                    2. [Vectorize](https://www.geeksforgeeks.org/vectorization-in-python/) the algorithm/code to the maximum possible extent because Numpy and OpenCV 
                    are optimized for vector operations.
                    3. Exploit the [cache coherence](https://www.geeksforgeeks.org/cache-coherence/).
                    4. Never make copies of array unless it is needed. 
                    Try to use views instead. Array copying is a costly operation.
                    Even after doing all these operations, if your code is still slow, or use of large loops are inevitable, use additional libraries like Cython to make it faster.
                    """)
        with st.expander("Using array view", expanded=False):
            st.code("""
                    from numpy import array

                    # creating an array
                    my_array = array([1, 2, 3, 4, 5])

                    # using the array.view() method
                    new_array = my_array.view()

                    # changing the 0 index of the array
                    new_array[0] = 9
                    print(my_array)
                    print(new_array)
                    """)
        
        st.markdown("""
                    Additional Resources
                    1. [Python Optimization Techniques](http://wiki.python.org/moin/PythonSpeed/PerformanceTips)
                    2. Scipy Lecture Notes - [Advanced Numpy](http://scipy-lectures.github.io/advanced/advanced_numpy/index.html#advanced-numpy)
                    3. [Timing and Profiling in IPython](http://pynash.org/2013/03/06/timing-and-profiling.html)
                    """)
        
        st.info("""
                ⚠️ **Note**
                > We are going to cover these topics with examples in the next updates.
                As of now, the official documents conclude like this.
                """)
        
class ImageProcessing(CommonComponents):
    def __init__(self):
        self.img = read_image("app/assets/Images/OpenCV_Logo_with_text.png") 
        self.img_file=None
        self.img_file_name = 'OpenCV_Logo_with_text.png'
    
class ChangingColorSpace(ImageProcessing):
    
    def Changing_Colorspace(self):
        st.markdown("""
                    There are more than 150 color-space conversion methods available in OpenCV. But we will look into only two which are most widely used ones, BGR \leftrightarrow Gray and BGR \leftrightarrow HSV.

                    For color conversion, we use the function `cv2.cvtColor(input_image, flag)` where flag determines the type of conversion.

                    For BGR $$\\rightarrow$$ Gray conversion we use the flags `cv2.COLOR_BGR2GRAY`. Similarly for BGR $$\\rightarrow$$ HSV, we use the flag `cv2.COLOR_BGR2HSV`. To get other flags, just run following commands in your Python terminal :
                    """)
        
        st.code("""
                >>> import cv2
                >>> flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
                >>> print flags
                """)
        
        with st.expander("Output", expanded=False):
            st.write("Click ':violet[▶️]' to see details")
            st.write(colorspace_flags())
        
        st.info("""
                ⚠️ **Note:**
                > For HSV, Hue range is [0,179], Saturation range is
                [0,255] and Value range is [0,255]. Different softwares
                use different scales. So if you are comparing OpenCV 
                values with them, you need to normalize these ranges.
                """)
    
    def Object_Tracking(self):
        st.markdown("""
                    Now we know how to convert BGR image to HSV, we can use this to extract a colored object. In HSV, it is more easier to represent a color than RGB color-space. In our application, we will try to extract a blue colored object. So here is the method:

                    - Take each frame of the video
                    - Convert from BGR to HSV color-space
                    - We threshold the HSV image for a range of blue color
                    - Now extract the blue object alone, we can do whatever on that image we want.

                    Below is the code which are commented in detail :
                    """)
        
        colorspaces = {
            
            "blue" : [[110,50,50], 
                      [130,255,255]],
            "green" : [[40, 40, 40],
                       [80, 255, 255]],
            "red" : [[0, 100, 100],
                     [10, 255, 255]]
            
        }
        
        colorspace = st.selectbox(label="Choose colorspace", options=["blue",
                                                                      "green",
                                                                      "red"])
        st.code(f"""
                import cv2
                import numpy as np

                cap = cv2.VideoCapture(0)

                while(1):

                    # Take each frame
                    _, frame = cap.read()

                    # Convert BGR to HSV
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # define range of blue color in HSV
                    lower_{colorspace} = np.array({colorspaces[colorspace][0]})
                    upper_{colorspace} = np.array({colorspaces[colorspace][1]})

                    # Threshold the HSV image to get only blue colors
                    mask = cv2.inRange(hsv, lower_{colorspace}, upper_{colorspace})

                    # Bitwise-AND mask and original image
                    res = cv2.bitwise_and(frame,frame, mask= mask)

                    cv2.imshow('frame',frame)
                    cv2.imshow('mask',mask)
                    cv2.imshow('res',res)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                cv2.destroyAllWindows()
                """)
        
        st.write(f"Below image shows tracking of the {colorspace} object:")
        frame, mask, res = object_tracking(self.img, colorspaces[colorspace])
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.image(frame, 'Original Frame', channels='BGR')
            col2.image(mask, 'Mask')
            col3.image(res, f'Tracked {colorspace} color', channels='BGR')

        st.info("""
                ⚠️ **Note**
                > There are some noises in the image. 
                We will see how to remove them in later chapters.
                """)
        
        st.info("""
                ⚠️ **Note**
                
                > This is the simplest method in object tracking.
                Once you learn functions of contours, you can do
                plenty of things like find centroid of this object
                and use it to track the object, draw diagrams just
                by moving your hand in front of camera and many other
                funny stuffs.
                """)
        
        st.write("""
                 ## How to find HSV values to track?
                This is a common question found in stackoverflow.com.
                It is very simple and you can use the same function,
                `cv2.cvtColor()`. 
                Instead of passing an image, you just pass the BGR values
                you want. For example, to find the HSV value of Green,
                try following commands in Python terminal:
                 """)
        with st.container(border=True):
            color = st.color_picker("Pick a color",value="#ff0000")
            color = list(ImageColor.getcolor(f'{color}','RGB')[::-1])
            st.code(f"""
                    >>> color = np.uint8([[{color}]])
                    >>> hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
                    >>> print hsv
                    {find_hsv_values(color)}
                    """)
        
        st.write("""
                 Now you take `[H-10, 100,100]` and `[H+10, 255, 255]` 
                 as lower bound and upper bound respectively. 
                 Apart from this method, you can use any image 
                 editing tools like GIMP or any online converters 
                 to find these values, but don’t forget to adjust 
                 the HSV ranges.
                 
                 ## Additional Resources
                 ### Exercises
                Try to find a way to extract more than one colored 
                objects, for eg, extract red, blue, green objects 
                simultaneously.
                 """)
        
        image = blank_image(*get_shape(self.img))
        for i in colorspaces:
            image+=object_tracking(self.img, colorspaces[i])[-1]
        with st.container(border=True):
            _, col, _ = st.columns(3)
            col.image(image, 'Expected Ouput', channels='BGR', width=200)
            
        with st.expander("Reveal Solution", expanded=False):
            st.code(f"""
            import numpy as np
            import cv2 as cv   
            def blank_image(height, width, channel):
                img = np.zeros((height, width, channel), np.uint8)
                return img
                
            def object_tracking(frame, colorspaces):
                # Convert BGR to HSV
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                # define range of blue color in HSV
                lower, upper = np.array(colorspaces[0]), np.array(colorspaces[1])

                # Threshold the HSV image to get only blue colors
                mask = cv.inRange(hsv, lower, upper)

                # Bitwise-AND mask and original image
                res = cv.bitwise_and(frame,frame, mask= mask)

                return frame, mask, res
                
            img = cv.imread("{self.img_file_name}")
            shape = img.shape
            blank_image = blank_image(*shape)
            colorspaces = {{
        
                "blue" : [[110,50,50], 
                        [130,255,255]],
                "green" : [[40, 40, 40],
                        [80, 255, 255]],
                "red" : [[0, 100, 100],
                        [10, 255, 255]]
                
            }}
            for i in colorspaces:
                blank_image+=object_tracking(img, colorspaces[i])[-1]
            cv.imshow('Expected Output', blank_image)
            cv.waitKey(0)
                """)

class GeometricTransformations(CommonComponents):
    
    def __init__(self):
        self.img = read_image(f"app/assets/Images/messi5.jpg")
        self.img_file=None
        self.img_file_name = 'messi5.jpg'
    
    def Scaling(self):
        st.markdown("""
                    Scaling is just resizing of the image. 
                    OpenCV comes with a function `cv2.resize()` for this purpose.
                    The size of the image can be specified manually, or you can specify
                    the scaling factor. Different interpolation methods are used. 
                    Preferable interpolation methods are `cv2.INTER_AREA` for shrinking 
                    and `cv2.INTER_CUBIC` (slow) & `cv2.INTER_LINEAR` for zooming. 
                    By default, interpolation method used is `cv2.INTER_LINEAR` 
                    for all resizing purposes. 
                    You can resize an input image either of following methods:
                    """)
        
        with st.container(border=True):
            st.subheader("Parameters")
            fx, fy = st.slider(label="fx", 
                            min_value=1, 
                            value=2, 
                            max_value=5),\
                    st.slider(label="fy",
                            min_value=1,
                            value=2,
                            max_value=5)
                    
            interpolations = st.selectbox(label="Interpolations:", 
                                        options=["INTER_CUBIC",
                                                    "INTER_AREA",
                                                    "INTER_LINEAR"])
        
            st.subheader("Code")
            st.code(f"""
                    import cv2
                    import numpy as np

                    img = cv2.imread('{self.img_file_name}')

                    res = cv2.resize(img,None,fx={fx}, fy={fy}, interpolation = cv2.{interpolations})

                    #OR

                    height, width = img.shape[:2]
                    res = cv2.resize(img,({fx}*width, {fy}*height), interpolation = cv2.{interpolations})
                    """)
            st.subheader("Output")
            st.image(scaling(self.img, fx, fy, interpolations), 
                         channels="BGR", use_column_width=True, 
                         caption=f"Output: {interpolations}")
        
    def Translation(self):
        st.markdown("""
                    Translation is the shifting of object’s location. 
                    If you know the shift in $$(x,y)$$ direction, let it be $$(t_x,t_y)$$, 
                    you can create the transformation matrix $$\textbf{M}$$ as follows:
                    """)
        
        st.latex(r"""
                 M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y  \end{bmatrix}
                 """)
        
        st.markdown("""
                    You can take make it into a Numpy array of type 
                    `np.float32` and pass it into `cv2.warpAffine()` function. 
                    See below example for a shift of `(100,50)`:
                    """)
        
        with st.container(border=True):
            st.subheader("Parameters")
            shift=st.slider("Shift: ", min_value=10, step=10, value=50, max_value=90)
            st.subheader("Code")
            st.code(f"""
                    import cv2
                    import numpy as np

                    img = cv2.imread('{self.img_file_name}',0)
                    rows,cols = img.shape

                    M = np.float32([[1,0,100],[0,1,{shift}]])
                    dst = cv2.warpAffine(img,M,(cols,rows))

                    cv2.imshow('img',dst)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    """)
            
            st.warning("""
                    ⚠️ **Warning**
                    > Third argument of the `cv2.warpAffine()` function is the size of 
                    the output image, which should be in the form of **(width, height)**. 
                    Remember width = number of columns, and height = number of rows.
                    """)
            
            st.markdown("See the result below:")
            
            # output
            st.subheader("Output")
            col1, col2 = st.columns(2)
            col1.image(self.img, channels="BGR", caption = 'Original')
            col2.image(translation(self.img, shift), channels='BGR', caption="img")
            
        
    def Rotation(self):
        st.markdown("""
                    Rotation of an image for an angle $$\\theta$$ is 
                    achieved by the transformation matrix of the form
                    """)
        
        st.latex(r"""
                 M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta   \end{bmatrix}
                 """)
        
        st.markdown("""
                    But OpenCV provides scaled rotation with 
                    adjustable center of rotation
                    so that you can rotate at any location you prefer. 
                    Modified transformation matrix is given by
                    """)
        
        st.latex(r"""
            \begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot \text{center.x} -  \beta \cdot \text{center.y} \\ -\beta & \alpha & \beta \cdot \text{center.x} + (1- \alpha ) \cdot \text{center.y} \end{bmatrix}
            """)

        
        st.write("where:")
        
        st.latex(r"""
                 \begin{array}{l} \alpha =  scale \cdot \cos \theta , \\ \beta =  scale \cdot \sin \theta \end{array}
                 """)
        
        st.markdown("""
                    To find this transformation matrix, OpenCV provides a function, 
                    `cv2.getRotationMatrix2D`. Check below example which rotates the image by 
                    90 degree with respect to center without any scaling.
                    """)
        # widgets here
        with st.container(border=True):
            st.subheader("Parameters")
            rotate = st.slider("Rotation in degree: ", min_value=0, step=10, value=90, max_value=360)
            st.subheader("Code")
            st.code(f"""
                    img = cv2.imread('{self.img_file_name}',0)
                    rows,cols = img.shape

                    M = cv2.getRotationMatrix2D((cols/2,rows/2),{rotate},1)
                    dst = cv2.warpAffine(img,M,(cols,rows))
                    """)
            
            # outputs here
            st.subheader("Output")
            st.image(rotation(self.img, rotate), channels="BGR",
                     caption="Rotaion", use_column_width=True)
    
    def AffineTransformation(self):
        st.markdown("""
                    In affine transformation, all parallel lines in the original 
                    image will still be parallel in the output image. To find the 
                    transformation matrix, we need three points from input image and 
                    their corresponding locations in output image. 
                    Then `cv2.getAffineTransform` will create a 2x3 matrix which 
                    is to be passed to cv2.warpAffine.

                    Check below example, and also look at the points I selected 
                    (which are marked in Green color):
                    """)
        
        # widgets here
        
        st.code("""
                img = cv2.imread('drawing.png')
                rows,cols,ch = img.shape

                pts1 = np.float32([[50,50],[200,50],[50,200]])
                pts2 = np.float32([[10,100],[200,50],[100,250]])

                M = cv2.getAffineTransform(pts1,pts2)

                dst = cv2.warpAffine(img,M,(cols,rows))

                plt.subplot(121),plt.imshow(img),plt.title('Input')
                plt.subplot(122),plt.imshow(dst),plt.title('Output')
                plt.show()
                """)
        
        st.write("See the result: ")
        
        # Output here
        
    def PerspectiveTransform(self):
        st.markdown("""
                    For perspective transformation, you need a 3x3 transformation matrix.
                    Straight lines will remain straight even after the transformation. 
                    To find this transformation matrix, you need 4 points on the input image
                    and corresponding points on the output image. Among these 4 points, 
                    3 of them should not be collinear. Then transformation matrix can be
                    found by the function `cv2.getPerspectiveTransform`. 
                    Then apply `cv2.warpPerspective` with this 3x3 transformation matrix.

                    See the code below:
                    """)
        # widget here
        
        st.code("""
                img = cv2.imread('sudokusmall.png')
                rows,cols,ch = img.shape

                pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
                pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

                M = cv2.getPerspectiveTransform(pts1,pts2)

                dst = cv2.warpPerspective(img,M,(300,300))

                plt.subplot(121),plt.imshow(img),plt.title('Input')
                plt.subplot(122),plt.imshow(dst),plt.title('Output')
                plt.show()
                """)
        
        # ouput here