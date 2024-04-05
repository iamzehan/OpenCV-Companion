import time
import streamlit as st
from PIL import ImageColor
import matplotlib.pyplot as plt
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
    rotation,
    affine_transform,
    perspective_transform,
    simple_thresholding,
    adaptive_thresholding,
    otsus_binarization,
    conv2D,
    averaging,
    gaussian_blur,
    median_blur,
    bilateral_filter,
    erosion,
    dilation,
    opening,
    closing,
    morph_gradient,
    top_hat,
    black_hat,
    get_morph,
    get_structuring_element,
    img_gradient,
    Canny,
    low_reso,
    high_reso,
    laplacian_levels,
    image_blending,
    get_started_contours,
    get_flags,
    draw_contours,
    get_moments,
    get_centroid,
    get_contour_approx,
    get_cvx_hull)

# Getting Started with Images (Page - 2)

class CommonComponents:
    def __init__(self):
        pass
    
    def uploader(self, multiple=False, custom_msg=None):
        
        # File and name handling
        file = st.file_uploader("Upload an Image:",
                                        type=["PNG","JPG"], 
                                        label_visibility="collapsed",
                                        accept_multiple_files=multiple)
        message = st.empty()
        if not file:
            if multiple:
                if custom_msg:
                    message.error(custom_msg)
                else:
                    message.error("Upload images to see changes")
            else:
                if custom_msg:
                    message.error(custom_msg)
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
                
    def grid(self, num_rows, num_columns, titles, images, clamp=False):
        for row in range(num_rows):
            columns = st.columns(num_columns)
            for col in range(num_columns):
                index = row * num_columns + col
                try:
                    columns[col].image(images[index], channels='BGR',clamp=clamp, caption=titles[index], use_column_width=True)
                except:
                    columns[col].image(images[index], caption=titles[index], clamp=clamp, use_column_width=True)
                        
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
        with st.sidebar:
            self.uploader()
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
                
        with st.container():
            
            st.write("""
                    You can access a pixel value by its row and column coordinates. 
                    For BGR image, it returns an array of Blue, Green, Red values. 
                    For grayscale image, just corresponding intensity is returned.
                    """)
            
            main = st.empty()
            
            with st.container(border=True):
                st.subheader("Playground")
                dim1 = st.empty()
                dim2 = st.empty()
                self.uploader()
                img = self.img
                row_max, column_max, channels = get_shape(img)
                dimensions = dim1.slider("Row (Height): ", value=100, max_value=row_max),\
                            dim2.slider("Column (Width):", value=100, max_value=column_max)
            with main:
                self.main_body()
            st.code(f"""
                    >>> px = img[{dimensions[0]},{dimensions[1]}]
                    >>> print( px )
                    """)
            with st.expander("Output: "):
                st.markdown(load_by_pixels(img, dimensions)[0])
            
            color = st.selectbox("Get pixel by specific color?:",options=["Blue", "Green", "Red"])
            colors = {'Blue':[0, '🔵'], 'Green': [1, '🟢'], 'Red':[2, '🔴']}
            
            st.code(f"""
                    # accessing only {color} {colors[color][1]} pixel
                    {color.lower()} = img[{dimensions[0]},{dimensions[1]}, {colors[color][0]}]
                    print( {color.lower()} )
                    """)
            color_result = load_by_pixels(img, dimensions, colors[color][0])
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
                    # accessing {color.upper()} {colors[color][1]} value
                    img.item({dimensions[0]}, {dimensions[1]}, {colors[color][0]})
                    >> {color_result}
                    # modifying {color.upper()} {colors[color][1]}  value
                    img.itemset({dimensions[0]}, {dimensions[1]}, {modify_value})
                    img.item({dimensions[0]}, {dimensions[1]}, {colors[color][0]})
                    >> {modify_value}
                    """)

    def Accessing_Image_Properties(self):
        st.markdown("""
                    ## Accessing Image Properties
                    """)
        
        with st.container(border=True):
            st.subheader("Try it yourself:")
            self.uploader()
            
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
        
        st.markdown("## Image ROI")
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
        with st.container(border=True):
            st.subheader("Try it yourself:")
            self.uploader()
            
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
                with container_space.container(border=True):
                    self.grid(2, 3, titles=["ORIGINAL",
                                            "REPLICATE",
                                            "REFLECT",
                                            "REFLECT 101",
                                            "WRAP",
                                            "CONSTANT"
                                            ],
                              images=[img, *make_borders(img)])
                    st.success("Showing Results")
                    info.error("Press ❌ to exit")
                    
                if button_space.button("❌"):
                    return
 
        st.markdown(f"""
                    ## Making Borders for Images (Padding)
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
        
        with st.container(border=True):
            self.uploader()

        show_code(self.img_file_name)
        show_image(self.img)

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
        with st.container(border=True):
            st.subheader("Try it yourself: ")
            self.uploader(multiple=True, custom_msg="Upload two images to see changes")
        with st.expander("Example:", expanded=False):
            self.grid(1, 2, titles=['img1', 'img2'], images=[self.img1, self.img2])
            
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
            self.uploader(multiple=True)
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
            st.subheader("Try it yourself:")
            self.uploader()
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
        
        code_placeholder= st.empty()
        with st.container(border=True):
            st.subheader("Try it yourself:")
            self.uploader()
            
        code_placeholder.code(f"""
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
        
        with st.container(border=True):
            st.subheader("Code")
            code_placeholder = st.empty()
            
        with st.container(border=True):
            st.subheader("Try it yourself: ")
            color_dots= {'blue': '🔵', 'red':'🔴', 'green':'🟢'}
            colorspace = st.selectbox(label="Choose colorspace", options=["blue",
                                                                        "green",
                                                                        "red"])
            self.uploader()
            
        code_placeholder.code(f"""
                import cv2
                import numpy as np

                cap = cv2.VideoCapture(0)

                while(1):

                    # Take each frame
                    _, frame = cap.read()

                    # Convert BGR to HSV
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # define range of {colorspace} - {color_dots[colorspace]} color in HSV
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
        
        st.write(f"Below image shows tracking of the {colorspace} - {color_dots[colorspace]} object:")
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

class GeometricTransformations(ImageProcessing):
    
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
            
            st.write("Upload image:")
            self.uploader()
        
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
            st.write("Upload image:")
            self.uploader()
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
            img, dst = translation(self.img, shift)
            col1.image(img, caption = 'Original', use_column_width=True)
            col2.image(dst, caption="img", use_column_width=True)
            
        
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
            st.write("Upload image:")
            self.uploader()
            st.subheader("Code")
            st.code(f"""
                    img = cv2.imread('{self.img_file_name}',0)
                    rows,cols = img.shape

                    M = cv2.getRotationMatrix2D((cols/2,rows/2),{rotate},1)
                    dst = cv2.warpAffine(img,M,(cols,rows))
                    """)
            
            # outputs here
            st.subheader("Output")
            st.image(rotation(self.img, rotate),
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
        with st.container(border=True):
            code_placeholder = st.empty()
            
            with st.container(border=True):
                st.subheader("Try it yourself:")
                self.uploader()
                
            with code_placeholder:
                st.subheader("Code")
                st.code(f"""
                        img = cv2.imread('{self.img_file_name}')
                        rows,cols,ch = img.shape

                        pts1 = np.float32([[50,50],[200,50],[50,200]])
                        pts2 = np.float32([[10,100],[200,50],[100,250]])

                        M = cv2.getAffineTransform(pts1,pts2)

                        dst = cv2.warpAffine(img,M,(cols,rows))

                        plt.subplot(121),plt.imshow(img),plt.title('Input')
                        plt.subplot(122),plt.imshow(dst),plt.title('Output')
                        plt.show()
                        """)
            
            st.subheader("Output")
            col1, col2 = st.columns(2)
            col1.image( self.img, channels="BGR", caption="Input", use_column_width=True)
            col2.image(affine_transform(self.img), channels="BGR", caption="Output", use_column_width=True)
    
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

        if not self.img_file:
            self.img=read_image("app/assets/Images/sudoku.jpg")
        st.markdown("**pts1** values:")
        
        with st.container(border=True):
            st.subheader("Parameters")
            pt1 = st.slider("`1`", value=[56,65], max_value=500),\
                st.slider("`2`", value =[368,52], max_value=500),\
                    st.slider("`3`", value = [28, 387], max_value=500),\
                    st.slider("`4`", value = [389,390], max_value = 500)
            
            pts1=[list(pt) for pt in pt1]
            st.write("Upload image:")
            self.uploader()
            st.subheader("Code")
            st.code(f"""
                    img = cv2.imread('sudokusmall.png')
                    rows,cols,ch = img.shape

                    pts1 = np.float32({pts1})
                    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

                    M = cv2.getPerspectiveTransform(pts1,pts2)

                    dst = cv2.warpPerspective(img,M,(300,300))

                    plt.subplot(121),plt.imshow(img),plt.title('Input')
                    plt.subplot(122),plt.imshow(dst),plt.title('Output')
                    plt.show()
                    """)
            
            # ouput here
            st.subheader("Output")
            col1, col2 = st.columns(2)
            col1.image(self.img, caption="Input", channels="BGR")
            col2.image(perspective_transform(self.img, pts1), caption="Output", channels="BGR")

class ImageThresholding(ImageProcessing):    
        
    def Simple_Thresholding(self):
        st.markdown("""
                    Here, the matter is straight forward. If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). The function used is cv2.threshold. First argument is the source image, which should be a grayscale image. Second argument is the threshold value which is used to classify the pixel values. Third argument is the maxVal which represents the value to be given if pixel value is more than (sometimes less than) the threshold value. OpenCV provides different styles of thresholding and it is decided by the fourth parameter of the function. Different types are:

                    - `cv2.THRESH_BINARY`
                    - `cv2.THRESH_BINARY_INV`
                    - `cv2.THRESH_TRUNC`
                    - `cv2.THRESH_TOZERO`
                    - `cv2.THRESH_TOZERO_INV`
                    
                    Documentation clearly explain what each type is meant for. Please check out the documentation.

                    Two outputs are obtained. First one is a retval which will be explained later. Second output is our thresholded image.
                    """)
        
        with st.container(border=True):
            st.subheader("Code")
            code_placeholder = st.empty()
            
        with st.container(border=True):
            st.subheader("Try it yourself: ")
            self.uploader()
            
        code_placeholder.code(f"""
                    import cv2
                    import numpy as np
                    from matplotlib import pyplot as plt

                    img = cv2.imread('{self.img_file_name}',0)
                    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
                    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
                    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
                    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

                    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
                    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

                    for i in xrange(6):
                        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
                        plt.title(titles[i])
                        plt.xticks([]),plt.yticks([])

                    plt.show()
                """)
        st.info("""
                ⚠️ Note

                > To plot multiple images, we have used `plt.subplot()` function. 
                Please checkout [Matplotlib docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) for more details.
                """)
        
        st.subheader("Output")
        with st.container(border=True):
            titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
            images = simple_thresholding(self.img)

            num_columns = 3
            num_rows = len(titles) // num_columns
            self.grid(num_rows, num_columns, titles, images)
            


        
            
    def Adaptive_Thresholding(self):
        st.markdown("""
                    In the previous section, we used one global value as a threshold. But this might not be good in all cases, 
                    e.g. if an image has different lighting conditions in different areas. In that case, adaptive thresholding can help. 
                    Here, the algorithm determines the threshold for a pixel based on a small region around it. So we get different thresholds 
                    for different regions of the same image which gives better results for images with varying illumination.

                    In addition to the parameters described above, the method `cv.adaptiveThreshold` takes three input parameters:

                    - The `adaptiveMethod` decides how the threshold value is calculated:
                    - `cv.ADAPTIVE_THRESH_MEAN_C`: The threshold value is the mean of the neighborhood area minus the constant C.
                    - `cv.ADAPTIVE_THRESH_GAUSSIAN_C`: The threshold value is a Gaussian-weighted sum of the neighborhood values minus the constant C.

                    - The `blockSize` determines the size of the neighborhood area and `C` is a constant that is subtracted from the mean or weighted sum of the neighborhood pixels.

                    The code below compares global thresholding and adaptive thresholding for an image with varying illumination:

                    """)
        if not self.img_file:
            self.img = read_image("app/assets/Images/sudoku.jpg")
            self.img_file_name='sudoku.jpg'
        
        with st.container(border=True):
            st.subheader("Code")
            code_placeholder = st.empty()
            
        with st.container(border=True):
            st.subheader("Try it yourself: ")
            self.uploader()
            
        code_placeholder.code(f"""
                    import cv2 as cv
                    import numpy as np
                    from matplotlib import pyplot as plt
                    
                    img = cv.imread('{self.img_file_name}', cv.IMREAD_GRAYSCALE)
                    assert img is not None, "file could not be read, check with os.path.exists()"
                    img = cv.medianBlur(img,5)
                    
                    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
                    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
                    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
                    
                    titles = ['Original Image', 'Global Thresholding (v = 127)',
                                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
                    images = [img, th1, th2, th3]
                    
                    for i in range(4):
                        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
                        plt.title(titles[i])
                        plt.xticks([]),plt.yticks([])
                    plt.show()
            """)
            
        st.subheader("Output")
        with st.container(border=True):
            titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
            images = adaptive_thresholding(self.img)
            num_columns = 2
            num_rows = len(titles) // num_columns
            self.grid(num_columns, num_rows, titles, images)
                
    def Otsus_Binarization(self):
        st.markdown("""
                    In global thresholding, we used an arbitrary chosen value as a threshold. 
                    In contrast, Otsu's method avoids having to choose a value and determines it automatically.

                    Consider an image with only two distinct image values (bimodal image), 
                    where the histogram would only consist of two peaks. 
                    A good threshold would be in the middle of those two values. 
                    Similarly, Otsu's method determines an optimal global threshold value from the image histogram.

                    In order to do so, the `cv.threshold()` function is used, where `cv.THRESH_OTSU` is passed as an 
                    extra flag. The threshold value can be chosen arbitrarily. The algorithm then finds the optimal 
                    threshold value which is returned as the first output.

                    Check out the example below. The input image is a noisy image. 
                    In the first case, global thresholding with a value of 127 is applied. 
                    In the second case, Otsu's thresholding is applied directly. 
                    In the third case, the image is first filtered with a 5x5 Gaussian kernel to remove the noise,
                    then Otsu thresholding is applied. See how noise filtering improves the result.
                    """)
        
        if not self.img_file:
            self.img_file_name = 'noisy.jpeg'
            self.img = read_image('app/assets/Images/noisy.jpeg')

        with st.container(border=True):
            st.subheader("Code")
            code_placeholder = st.empty()
        
        with st.container(border=True):
            st.subheader("Try it yourself: ")
            self.uploader()
            
        code_placeholder.code(f"""
                import cv2 as cv
                import numpy as np
                from matplotlib import pyplot as plt
                
                img = cv.imread('{self.img_file_name}', cv.IMREAD_GRAYSCALE)
                assert img is not None, "file could not be read, check with os.path.exists()"
                
                # global thresholding
                ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
                
                # Otsu's thresholding
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                
                # Otsu's thresholding after Gaussian filtering
                blur = cv.GaussianBlur(img,(5,5),0)
                ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                
                # plot all the images and their histograms
                images = [img, 0, th1,
                        img, 0, th2,
                        blur, 0, th3]
                titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                        'Original Noisy Image','Histogram',"Otsu's Thresholding",
                        'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
                
                for i in range(3):
                    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
                    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
                    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
                    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
                    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
                    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
                plt.show()
                """)
        st.subheader("Output")
        
        with st.container(border=True):
            titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                        'Original Noisy Image','Histogram',"Otsu's Thresholding",
                        'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            images=otsus_binarization(self.img)
            
            # 3 rows
            for i in range(3):
                axes[i, 0].imshow(images[i * 3], cmap='gray')
                axes[i, 0].set_title(titles[i * 3])
                axes[i, 0].axis('off')

                axes[i, 1].hist(images[i * 3].ravel(), 256)
                axes[i, 1].set_title(titles[i * 3 + 1])

                axes[i, 2].imshow(images[i * 3 + 2], cmap='gray')
                axes[i, 2].set_title(titles[i * 3 + 2])
                axes[i, 2].axis('off')
            st.pyplot(fig)
                
        st.markdown("""
                        #### How does Otsu's Binarization work?
                        This section demonstrates a Python implementation of Otsu's binarization to show how it actually works.
                        If you are not interested, you can skip this.

                        Since we are working with bimodal images, Otsu's algorithm tries to find a threshold value (t) which minimizes 
                        the weighted within-class variance given by the relation:
                        """)
        with st.expander("Reveal Equations", expanded=False):
            st.latex(r"\sigma_w^2(t) = q_1(t)\sigma_1^2(t)+q_2(t)\sigma_2^2(t)")
            
            st.markdown("where")
            
            st.latex(r"""q_1(t) = \sum_{i=1}^{t} P(i) \quad \& \quad q_2(t) = \sum_{i=t+1}^{I} P(i) \\
                    \mu_1(t) = \sum_{i=1}^{t} \frac{iP(i)}{q_1(t)} \quad \& \quad \mu_2(t) = \sum_{i=t+1}^{I} \frac{iP(i)}{q_2(t)} \\
                        \sigma_1^2(t) = \sum_{i=1}^{t} [i-\mu_1(t)]^2 \frac{P(i)}{q_1(t)} \quad \& \quad \sigma_2^2(t) = \sum_{i=t+1}^{I} [i-\mu_2(t)]^2 \frac{P(i)}{q_2(t)}""")
        
        st.write("""It actually finds a value of t which lies in between two peaks such that variances to both classes are minimal.
                 It can be simply implemented in Python as follows:""")
        
        st.code("""
                img = cv.imread('noisy2.png', cv.IMREAD_GRAYSCALE)
                assert img is not None, "file could not be read, check with os.path.exists()"
                blur = cv.GaussianBlur(img,(5,5),0)
                
                # find normalized_histogram, and its cumulative distribution function
                hist = cv.calcHist([blur],[0],None,[256],[0,256])
                hist_norm = hist.ravel()/hist.sum()
                Q = hist_norm.cumsum()
                
                bins = np.arange(256)
                
                fn_min = np.inf
                thresh = -1
                
                for i in range(1,256):
                    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
                    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
                    if q1 < 1.e-6 or q2 < 1.e-6:
                        continue
                    b1,b2 = np.hsplit(bins,[i]) # weights
                
                    # finding means and variances
                    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
                    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
                
                    # calculates the minimization function
                    fn = v1*q1 + v2*q2
                    if fn < fn_min:
                        fn_min = fn
                        thresh = i
                
                # find otsu's threshold value with OpenCV function
                ret, otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                print( "{} {}".format(thresh,ret) )
                """)
        
        st.markdown("""
                    ### Additional Resources
                    - [Digital Image Processing, Rafael C. Gonzalez](https://dl.icdst.org/pdfs/files4/01c56e081202b62bd7d3b4f8545775fb.pdf) 
                    #### Exercises
                    There are some optimizations available for Otsu's binarization. You can search and implement it.
                    * You can also look at this [link](https://learnopencv.com/otsu-thresholding-with-opencv/)
                    """) 

class SmoothingImages(ImageProcessing):
    
    def Convolution2D(self):
        
        st.markdown("""
                    As in one-dimensional signals, images also can be filtered with various low-pass filters (LPF),
                    high-pass filters (HPF), etc. LPF helps in removing noise, blurring images, etc. HPF filters help
                    in finding edges in images.

                    OpenCV provides a function `cv.filter2D()` to convolve a kernel with an image. As an example, we will
                    try an averaging filter on an image. A 5x5 averaging filter kernel will look like the below:
                    """)
        
        st.latex(r"K =  \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}")
        
        st.markdown("""
                    The operation works like this: keep this kernel above a pixel, add all the 25 pixels below this kernel,
                    take the average, and replace the central pixel with the new average value. This operation is continued 
                    for all the pixels in the image. Try this code and check the result:
                    """)
        st.subheader("Code")
        code_placeholder=st.empty()
        info1 = st.info("Example Code")
        
        with st.container(border=True):
            dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
            dim = dimensions[st.selectbox("Kernel Dimensions:", index=1, 
                                      options=["3 x 3", "5 x 5", "7 x 7"])]
            st.write("Upload image:")
            self.uploader()
            
        code_placeholder.code(f"""
                import numpy as np
                import cv2 as cv
                from matplotlib import pyplot as plt
                
                img = cv.imread('{self.img_file_name}')
                assert img is not None, "file could not be read, check with os.path.exists()"
                
                kernel = np.ones({dim},np.float32)/{dim[0]*dim[1]}
                dst = cv.filter2D(img,-1,kernel)
                
                plt.subplot(121),plt.imshow(img),plt.title('Original')
                plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
                plt.xticks([]), plt.yticks([])
                plt.show()
                """)
        
        images = [self.img, conv2D(self.img, dim, dim[0]*dim[1])]
        titles = ['Original', 'Averaging']
        
        st.subheader("Output")
        
        with st.container(border=True):
            self.grid(1, 2, titles, images)
            
        info2 = st.info("Example Output")
        if dim!=(5,5) or self.img_file:
            info1.success("Showing Changed Code")
            info2.success("Showing Changed Output")
        

    def ImageBlurring(self):
        
        def Averaging():
            st.subheader("1. Averaging")
            
            st.markdown("""
                        This is done by convolving an image with a normalized box filter.
                        It simply takes the average of all the pixels under the kernel 
                        area and replaces the central element. This is done by the function
                        `cv.blur()` or `cv.boxFilter()`. Check the docs for more details about 
                        the kernel. We should specify the width and height of the kernel. 
                        A `3x3` normalized box filter would look like the below:
                        """)
            
            st.latex(r"K =  \frac{1}{9} \begin{bmatrix} 1 & 1 & 1  \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}")
            
            st.info("""
                    ⚠️ Note
                    > If you don't want to use a normalized box filter, use `cv.boxFilter()`. 
                    Pass an argument normalize=False to the function.
                    """)
            
            sample = st.empty()
            
            code_placeholder=st.empty()
            info1 = st.info("Example Code")
            
            with st.container(border=True):
                dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
                dim = dimensions[st.selectbox("Kernel Dimensions:", index=1, 
                                        options=["3 x 3", "5 x 5", "7 x 7"])]
                st.write("Upload image:")
                self.uploader()
            
            sample.markdown(f"Check a sample demo below with a kernel of `{dim}` size:")
            code_placeholder.code(f"""
                    import cv2 as cv
                    import numpy as np
                    from matplotlib import pyplot as plt
                    
                    img = cv.imread('{self.img_file_name}')
                    assert img is not None, "file could not be read, check with os.path.exists()"
                    
                    blur = cv.blur(img,{dim})
                    
                    plt.subplot(121),plt.imshow(img),plt.title('Original')
                    plt.xticks([]), plt.yticks([])
                    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
                    plt.xticks([]), plt.yticks([])
                    plt.show()
                    """)
            
            st.subheader("Output")
            self.grid(1, 2, titles=['Original', 'Blurred'], 
                      images=[self.img, averaging(self.img, dim)])
            info2 = st.info("Example Output")
            
            if dim!=(5,5) or self.img_file:
                sample.markdown(f"Check the changed code below with a kernel of `{dim}` size:")
                info1.success("Showing Changed Code")
                info2.success("Showing Changed Output")
                
        def GaussianBlurring():
            st.subheader("2. Gaussian Blurring")
            st.markdown("""
                        In this method, instead of a box filter, a Gaussian kernel is used.
                        It is done with the function, `cv.GaussianBlur()`. 
                        We should specify the width and height of the kernel which should 
                        be positive and odd. We also should specify the standard deviation 
                        in the X and Y directions, sigmaX and sigmaY respectively. If only 
                        sigmaX is specified, sigmaY is taken as the same as sigmaX. If both 
                        are given as zeros, they are calculated from the kernel size. 
                        Gaussian blurring is highly effective in removing Gaussian noise 
                        from an image.

                        If you want, you can create a Gaussian kernel with the function, 
                        `cv.getGaussianKernel()`.

                        The previous code can be modified for Gaussian blurring:
                        """)
            
            dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
            
            col1, col2 = st.columns([4, 8])
            
            with col1.container(border=True):
                st.subheader("Parameters")
                dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
                intensity = st.slider('Intensity', max_value=10)
                
                
            with col2.container(border=True):
                st.subheader("Code")
                st.divider()
                st.code(f"blur = cv.GaussianBlur(img,{dim},{intensity})")
                self.uploader()
            
            info1=st.info("Example code")
            st.subheader("Output")
            
            self.grid(1, 2, titles=['Original', 'Blurred'], images=[self.img, gaussian_blur(self.img, dim, intensity)])
            info2= st.info("Example Output")
            
            if self.img_file or dim!=(5,5) or intensity!=0:
                info1.success("Changed code")
                info2.success("Changed Output")
                
            
        def MedianBlurring():
            st.subheader("3. Median Blurring")
            
            st.markdown("""
            The `cv.medianBlur()` function takes the median of all the pixels under
            the kernel area, and the central element is replaced with this median value. 
            This is highly effective against salt-and-pepper noise in an image. Interestingly,
            in the above filters, the central element is a newly calculated value, 
            which may be a pixel value in the image or a new value. However, in median blurring,
            the central element is always replaced by some pixel value in the image, 
            reducing the noise effectively. The kernel size should be a positive odd integer.

            In this demo, I added a 50% noise to our original image and applied median blurring. 
            Check the result:
            """)
            
            col1, col2 = st.columns([3, 9])
            
            with col1.container(border=True):
                st.subheader("Parameter")
                intensity = st.slider('Intensity', min_value=1, value=5, step=2, max_value=9)
            with col2.container(border=True):
                st.subheader("Code")
                st.code(f"median = cv.medianBlur(img,{intensity})")
                self.uploader()
            
            
            st.subheader("Output")
            self.grid(1, 2, titles=['Original', 'Median Blur'], images = [self.img, median_blur(self.img, intensity)])

        
        def BilateralFiltering():
            st.subheader("4. Bilateral Filtering")

            st.markdown("""
            ### Bilateral Filtering

            The `cv.bilateralFilter()` is highly effective in noise removal while preserving sharp edges, though it's slower compared to other filters.

            We already saw that a Gaussian filter takes the neighborhood around the pixel and finds its Gaussian weighted average. 
            This filter depends solely on space, considering nearby pixels during filtering. 
            However, it doesn't account for similar pixel intensities or distinguish between edge and non-edge pixels, resulting in edge blurring.

            Bilateral filtering employs two Gaussian filters: one for space and another for pixel intensity difference. 
            The space filter considers only nearby pixels for blurring, while the intensity difference filter considers only pixels with similar intensities. 
            This dual approach preserves edges, as pixels at edges typically exhibit large intensity variations.

            The below sample demonstrates the use of a bilateral filter. (For details on arguments, expand the Details):
            """)
            with st.expander("Details:"):
                st.write("""
                         The `cv.bilateralFilter()` function in OpenCV is used for bilateral filtering, and it has the following parameters:

                        `img`: This is the input image. It should be a matrix of type uint8 or float32, representing the image to be filtered.

                        `d`: Diameter of each pixel neighborhood. It is an integer representing the size of the pixel neighborhood used during filtering.
                        A larger value of d means that more distant pixels will influence each other during filtering.

                        `sigmaColor`: Standard deviation in the color space. This parameter controls how different colors will be considered as similar.
                        A smaller value of sigmaColor means that only pixels with very similar colors will be averaged for filtering.

                        `sigmaSpace`: Standard deviation in the coordinate space. This parameter controls how far away pixels will influence each other in the spatial domain. 
                        A smaller value of sigmaSpace means that only pixels within a close spatial proximity will be considered for filtering.

                        In our example 
                        ```python 
                        blur = cv.bilateralFilter(img, 9, 75, 75)
                        ```
                        
                        it means:

                        img: The input image.
                        9: The diameter of each pixel neighborhood. In this case, it's set to 9.
                        75: The standard deviation in the color space (sigmaColor). It controls color similarity.
                        75: The standard deviation in the coordinate space (sigmaSpace). It controls spatial proximity.
                         """)
                
            with st.container(border=True):
                st.subheader("Parameters")
                d = st.slider("Diameter of Pixel Neighborhood (d)", 1, 20, 9, 1)
                sigma_color = st.slider("Sigma Color", 1, 200, 75, 1)
                sigma_space = st.slider("Sigma Space", 1, 200, 75, 1)
                self.uploader()
            
            st.subheader("Code")
            st.code(f"blur = cv.bilateralFilter(img, {d}, {sigma_color}, {sigma_space})")
            
            st.subheader("Output")
            self.grid(1, 2, titles=['Original', 'Bilateral Filter'], images=[self.img, bilateral_filter(self.img, d, sigma_color, sigma_space)])
            
            st.write("See, the texture on the surface is gone, but the edges are still preserved.")
            
        st.markdown("""
                    ### Additional Resources
                    - Details about the [bilateral filtering](https://people.csail.mit.edu/sparis/bf_course/)
                    """)
        
        st.markdown("""
                    Image blurring is achieved by convolving the image with 
                    a low-pass filter kernel. It is useful for removing noise.
                    It actually removes high frequency content (eg: noise, edges)
                    from the image. So edges are blurred a little bit in this operation 
                    (there are also blurring techniques which don't blur the edges). 
                    OpenCV provides four main types of blurring techniques.
                    """)
        
        options = st.radio("Options: ", 
                            options = [
                                "Averaging",
                                "Gaussian Blurring",
                                "Median Blurring",
                                "Bilateral Filtering"
                                ],
                            horizontal=True,
                    label_visibility="collapsed")
        
        with st.container(border=True):
            if options == "Averaging":
                Averaging()
                
            elif options == "Gaussian Blurring":
                GaussianBlurring()
                
            elif options == "Median Blurring":
                MedianBlurring()
                
            elif options == "Bilateral Filtering":
                BilateralFiltering()

class MorphologicalTransformation(ImageProcessing):
    def __init__(self):
        self.img = read_image("app/assets/Images/j.png")
        self.img_file_name = 'j.png'
        self.img_file=None
    
    def Theory(self):
        st.markdown("""
                    ### Theory
                    Morphological transformations are some simple operations based on the image shape. 
                    It is normally performed on binary images. It needs two inputs, one is our original image,
                    second one is called structuring element or kernel which decides the nature of operation.
                    Two basic morphological operators are Erosion and Dilation. Then its variant forms like Opening,
                    Closing, Gradient etc also comes into play. We will see them one-by-one with help of following image:
                    """)
        st.image("app/assets/Images/j.png", width=200)
    
    def Erosion(self):
        st.subheader("1. Erosion")
        st.markdown("""
                    The basic idea of erosion is similar to soil erosion. It erodes away the boundaries of the foreground object 
                    (Always try to keep foreground in white). So, what does it do? The kernel slides through the image (as in 2D 
                    convolution). A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under
                    the kernel are 1; otherwise, it is eroded (made to zero).

                    So what happens is that all the pixels near the boundary will be discarded depending upon the size of the kernel.
                    The thickness or size of the foreground object decreases, or simply, the white region decreases in the image. 
                    It is useful for removing small white noises (as we have seen in the color space chapter), detaching two connected
                    objects, etc.

                    Here, as an example, I would use a 5x5 kernel filled with ones. Let’s see how it works:
                    """
                    )
        
        st.subheader("Perameters")
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
        iterations = st.slider("Iterations: ", min_value=1, value=1, max_value=10)
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code(f"""
                import cv2
                import numpy as np

                img = cv2.imread('{self.img_file_name}',0)
                kernel = np.ones({dim},np.uint8)
                erosion = cv2.erode(img, kernel, iterations = {iterations})
                """)
        st.subheader("Output")
        images = [self.img, erosion(self.img, dim, iterations)]
        titles = ['Original', 'Erosion']
        self.grid(1, 2, titles, images)
        st.info("""
                ⚠️ Note:
                > In order for the function `cv2.erode()` to work, the input image must be
                a binary image or to be converted into a binary image. We can do that by 
                using the following function:
                
                ```python
                def get_binary_image(img):
                    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    return binary_image
                ```
                """)
    def Dilation(self):
        st.subheader("2. Dilation")
        st.markdown("""
                    It is just opposite of erosion. Here, a pixel element is ‘1’ if at least one pixel under the kernel is ‘1’. 
                    So it increases the white region in the image or the size of the foreground object increases. Normally, 
                    in cases like noise removal, erosion is followed by dilation. Because erosion removes white noises, 
                    but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, 
                    but our object area increases. It is also useful in joining broken parts of an object.
                    """
                    )
        st.subheader("Perameters")
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
        iterations = st.slider("Iterations: ", min_value=1, value=1, max_value=10)
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code(f"""
                dilation = cv2.dilate(img,kernel,iterations = {iterations})
                """)
        erode, dilate = dilation(self.img, dim, iterations)
        st.subheader("Output")
        self.grid(1, 3, titles=['Original','Erosion', 'Dilation'],images=[self.img, erode, dilate])
        
    def Opening(self):
        st.subheader("3. Opening")
        st.markdown("""
                    Opening is just another name of **erosion followed by dilation**. 
                    It is useful in removing noise, 
                    as we explained above. Here we use the function, `cv2.morphologyEx()`
                """)
        if not self.img_file:
            self.img = read_image('app/assets/Images/j - noise.png')
        st.subheader("Perameters")
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code("""
                opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                """)
        st.subheader("Output")
        self.grid(1, 2, titles=['Original', 'Opening'], images=[self.img, opening(self.img, dim)])
    
    def Closing(self):
        st.subheader("4. Closing")
        st.markdown("""
                    Closing is reverse of Opening, Dilation followed by Erosion. 
                    It is useful in closing small holes inside the foreground objects, 
                    or small black points on the object.
                """)
        
        if not self.img_file:
            self.img = read_image('app/assets/Images/j - holes.png')
            
        st.subheader("Perameters")
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code("""
                closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                """)
        st.subheader("Output")
        self.grid(1, 2, titles=['Original', 'Closing'], images=[self.img, closing(self.img, dim)])
        
    def MorphGradient(self):
        st.subheader("5. Morphological Gradient")
        st.markdown("""
                    It is the difference between dilation and erosion of an image.

                    The result will look like the outline of the object.
                """)
        st.subheader("Perameters")
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7"])]
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code("""
                gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
                """) 
        st.subheader("Output")
        self.grid(1, 2, titles=['Original', 'Morphological Gradient'], images = [self.img, morph_gradient(self.img, dim)])
        
    def TopHat(self):
        st.subheader("6. Top Hat")
        st.markdown("""
                    It is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel.
                """)
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7), "9 x 9": (9, 9)}
        dim = dimensions[st.selectbox("Kernel:", index=3, options=["3 x 3", "5 x 5", "7 x 7", "9 x 9"])]
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code("""
                tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
                """)
        st.subheader("Output")
        self.grid(1, 2, titles=['Original', 'Top Hat'], images = [self.img, top_hat(self.img, dim)])
          
    def BlackHat(self):
        st.subheader("7. Black Hat")
        st.markdown("""
                    It is the difference between the closing of the input image and input image.
                """)
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7), "9 x 9": (9, 9)}
        dim = dimensions[st.selectbox("Kernel:", index=3, options=["3 x 3", "5 x 5", "7 x 7", "9 x 9"])]
        st.write("Upload image: ")
        self.uploader()
        
        st.subheader("Code")
        st.code("""
                blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
                """)
        st.subheader("Output")
        self.grid(1, 2, titles=['Original', 'Black Hat'], images = [self.img, black_hat(self.img, dim)])
        
    def StructuringElement(self):
        st.subheader("Structuring Element")
        st.markdown("""
                    We manually created a structuring elements in the previous examples with help of Numpy.
                    It is rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. 
                    So for this purpose, OpenCV has a function, `cv2.getStructuringElement()`. 
                    You just pass the shape and size of the kernel, you get the desired kernel.
                """)
        option = st.selectbox("Options:", index=2, options=get_morph())
        dimensions = {"3 x 3": (3, 3), "5 x 5": (5,5), "7 x 7": (7, 7), "9 x 9": (9, 9)}
        dim = dimensions[st.selectbox("Kernel:", index=1, options=["3 x 3", "5 x 5", "7 x 7", "9 x 9"])]
        st.subheader("Code")
        st.code(f"""
                import cv2
                cv2.getStructuringElement(cv2.{option},{dim})
                """)
        st.subheader("Output")
        try:
            st.code(f"{get_structuring_element(option, dim)}")
        except:
            st.info(f"Option not available for {option}")

class ImageGradients(ImageProcessing):

    def Theory(self, param):
        st.markdown("""
                    # Gradient Filters in OpenCV

                    OpenCV provides three types of gradient filters or High-pass filters: Sobel, Scharr, and Laplacian. 
                    We will explore each one of them.

                    ## 1. Sobel and Scharr Derivatives

                    Sobel operators are a joint Gaussian smoothing plus differentiation operation, making them more resistant to noise. 
                    You can specify the direction of derivatives to be taken (vertical or horizontal) using the arguments `yorder` and `xorder` respectively.
                    Additionally, you can specify the size of the kernel with the argument `ksize`. If `ksize = -1`, a 3x3 Scharr filter is used, which gives
                    better results than a 3x3 Sobel filter. Please refer to the documentation for details on the kernels used.

                    ## 2. Laplacian Derivatives

                    Laplacian Derivatives calculate the Laplacian of the image using the relation:
                    $$\Delta src = \\frac{\partial ^2{src}}{\partial x^2} + \\frac{\partial ^2{src}}{\partial y^2}$$ 
                    where each derivative is found using Sobel derivatives. If `ksize = 1`, the following kernel is used for filtering:""")

        st.latex(r"kernel = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0  \end{bmatrix}")
        
        code_placeholder = st.empty()
        st.write("Upload image: ")
        self.uploader()
        with code_placeholder:
            st.subheader("Code")
            st.markdown("""Below code shows all operators in a single diagram. 
                        All kernels are of 5x5 size. Depth of output image is 
                        passed -1 to get the result in np.uint8 type.""")
            
            if not self.img_file:
                self.img = read_image('app/assets/Images/sudoku.jpg')
                self.img_file_name = 'sudoku.jpg'
                
            st.code(f"""
                    import cv2
                    import numpy as np
                    from matplotlib import pyplot as plt

                    img = cv2.imread('{self.img_file_name}',0)

                    laplacian = cv2.Laplacian(img,cv2.CV_64F)
                    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
                    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

                    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
                    plt.title('Original'), plt.xticks([]), plt.yticks([])
                    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
                    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
                    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
                    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
                    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
                    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

                    plt.show()
                    """)
        st.subheader("Output")
        self.grid(2, 2, titles=["Original", "Laplacian", "Sobel - X", "Sobel - Y"],
                  images=img_gradient(self.img, param),clamp=True)
        
        if self.img_file:
            st.success("Showing results for uploaded image") 
        else:
            st.info("Showing results for example image")
    
    def Important(self, param):
        
        st.markdown("""
                    In our last example, the output datatype is `cv2.CV_8U` or `np.uint8`. 
                    However, there is a slight problem with that. The Black-to-White transition 
                    is taken as a positive slope (it has a positive value), while White-to-Black 
                    transition is taken as a negative slope (it has a negative value). So when you
                    convert data to `np.uint8`, all negative slopes are made zero. 
                    In simple words, you miss that edge.

                    If you want to detect both edges, a better option is to keep the output datatype 
                    to some higher forms, like `cv2.CV_16S`, `cv2.CV_64F`, etc., take its absolute value,
                    and then convert back to `cv2.CV_8U`. The below code demonstrates this procedure for 
                    a horizontal Sobel filter and highlights the difference in results.
                    """)
        
        if not self.img_file:
            self.img = read_image('app/assets/Images/box.png')
            self.img_file_name = 'box.png'
        
        code_placeholder = st.empty()
        st.write("Upload image: ")
        self.uploader()
        with code_placeholder:
            st.subheader("Code")
            st.code(f"""
                    import cv2
                    import numpy as np
                    from matplotlib import pyplot as plt

                    img = cv2.imread('{self.img_file_name}',0)

                    # Output dtype = cv2.CV_8U
                    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

                    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
                    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
                    abs_sobel64f = np.absolute(sobelx64f)
                    sobel_8u = np.uint8(abs_sobel64f)

                    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
                    plt.title('Original'), plt.xticks([]), plt.yticks([])
                    plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
                    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
                    plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
                    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

                    plt.show()
                    """)
        
        st.subheader("Output")
        self.grid(1, 3, titles=["Original", "Sobel CV_8U", "Sobel abs(CV_64F)"],
                  images=img_gradient(self.img, param),clamp=True)
        
        if self.img_file:
            st.success("Showing results for uploaded image") 
        else:
            st.info("Showing results for example image")

class CannyEdgeDetection(ImageProcessing):
    
    def Theory(self):
        st.markdown("""
                    ## Theory
                    Canny Edge Detection is a popular edge detection algorithm. It was developed by John F. Canny in 1986.
                    It is a multi-stage algorithm and we will go through each stages.

                    #### 1. Noise Reduction
                    Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image
                    with a 5x5 Gaussian filter. We have already seen this in previous chapters.

                    #### 2. Finding Intensity Gradient of the Image
                    Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction to get first
                    derivative in horizontal direction $$(G_x)$$ and vertical direction $$(G_y)$$. From these two images, we can 
                    find edge gradient and direction for each pixel as follows:
                    """)
        
        st.latex(r"Edge\_Gradient \; (G) = \sqrt{G_x^2 + G_y^2} \\ Angle \; (\theta) = \tan^{-1} \bigg(\frac{G_y}{G_x}\bigg)")
        
        st.markdown("""
                    Gradient direction is always perpendicular to edges. It is rounded to one of four angles representing vertical, 
                    horizontal and two diagonal directions.
                    
                    #### 3. Non-maximum Suppression
                    After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted pixels which 
                    may not constitute the edge.
                    For this, at every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient.
                    Check the image below:
                    """)
        
        st.image("app/assets/Images/nms.jpg", caption="Non-maximum Suppression", use_column_width=True)
        
        st.markdown("""
                    Point A is on the edge ( in vertical direction). Gradient direction is normal to the edge.
                    Point B and C are in gradient directions. So point A is checked with point B and C to see 
                    if it forms a local maximum. If so, it is considered for next stage, otherwise, it is suppressed ( put to zero).

                    In short, the result you get is a binary image with “thin edges”.
                    
                    #### 4. Hysteresis Thresholding
                    This stage decides which are all edges are really edges and which are not. For this, we need two threshold values, 
                    minVal and maxVal. Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal 
                    are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges 
                    based on their connectivity. If they are connected to “sure-edge” pixels, they are considered to be part of edges. 
                    Otherwise, they are also discarded. See the image below:
                    """)
        
        st.image("app/assets/Images/hysteresis.jpg", caption="Hysteresis Thresholding", use_column_width=True)
        
        st.markdown("""
                    The edge A is above the maxVal, so considered as “sure-edge”. Although edge C is below maxVal, it is connected to edge A, 
                    so that also considered as valid edge and we get that full curve. But edge B, although it is above minVal and is in same 
                    region as that of edge C, it is not connected to any “sure-edge”, so that is discarded. So it is very important that we 
                    have to select minVal and maxVal accordingly to get the correct result.

                    This stage also removes small pixels noises on the assumption that edges are long lines.

                    So what we finally get is strong edges in the image.
                    """)
    
    def Canny_Edge_Detection(self):
        st.markdown("""
                    Canny Edge Detection in OpenCV
                    OpenCV puts all the above in single function, cv2.Canny(). We will see how to use it. First argument is our input image. 
                    Second and third arguments are our minVal and maxVal respectively. Third argument is aperture_size. It is the size of 
                    Sobel kernel used for find image gradients. By default it is 3. Last argument is L2gradient which specifies the equation 
                    for finding gradient magnitude. 
                    If it is True, it uses the equation mentioned above which is more accurate, otherwise it uses this function: 
                    
                    $$Edge\_Gradient \; (G) = |G_x| + |G_y|$$ . By default, it is `False`.
                    """)
        
        if not self.img_file:
            self.img = read_image("app/assets/Images/messi5.jpg")
            self.img_file_name = 'messi5.jpg'
        
        st.subheader("Parameters")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            minVal, maxVal = col1.slider("`minVal`", value=100, max_value=500), col2.slider("`maxVal`", value=200, max_value=500)
            st.write("Upload image: ")
            self.uploader()
            
        st.subheader("Code")
        st.code(f"""
                import cv2
                import numpy as np
                from matplotlib import pyplot as plt

                img = cv2.imread('{self.img_file_name}',0)
                edges = cv2.Canny(img,{minVal},{maxVal})

                plt.subplot(121),plt.imshow(img,cmap = 'gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(edges,cmap = 'gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

                plt.show()
                """)

        st.subheader("Output")
        self.grid(1, 2, titles=["Original", "Canny Edge Detection"], images=Canny(self.img, minVal, maxVal))
        
        st.markdown("""
                    #### Additional Resources
                    1. Canny edge detector at [Wikipedia](http://en.wikipedia.org/wiki/Canny_edge_detector)
                    2. [Canny Edge Detection Tutorial](http://dasl.mem.drexel.edu/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html) by Bill Green, 2002.
                    """)
        
class ImagePyramids(ImageProcessing):
    
    def Introduction(self):
        st.subheader("Goals")
        st.markdown("""
                    In this chapter,
                    - We will learn about Image Pyramids
                    - We will use Image pyramids to create a new fruit, “Orapple”
                    - We will see these functions: `cv2.pyrUp()`, `cv2.pyrDown()`
                    """)
    
    def Theory(self):
        st.subheader("Theory")
        st.markdown("""
                    Normally, we used to work with an image of constant size. 
                    But in some occassions, we need to work with images of different
                    resolution of the same image. For example, while searching for 
                    something in an image, like face, we are not sure at what size 
                    the object will be present in the image. In that case, we will 
                    need to create a set of images with different resolution and search
                    for object in all the images. These set of images with different resolution
                    are called Image Pyramids (because when they are kept in a stack with 
                    biggest image at bottom and smallest image at top look like a pyramid).

                    There are two kinds of Image Pyramids. 1) Gaussian Pyramid and 2) Laplacian Pyramids

                    Higher level (Low resolution) in a Gaussian Pyramid is formed by removing consecutive
                    rows and columns in Lower level (higher resolution) image. Then each pixel in higher 
                    level is formed by the contribution from 5 pixels in underlying level with gaussian 
                    weights. By doing so, a $$M \\times N$$ image becomes $$M/2 \\times N/2$$ image. So area reduces 
                    to one-fourth of original area. It is called an Octave. The same pattern continues as 
                    we go upper in pyramid (ie, resolution decreases). Similarly while expanding, area becomes
                    4 times in each level. We can find Gaussian pyramids using `cv2.pyrDown()` and `cv2.pyrUp()` functions.
                    """)
                
        if not self.img_file:
            self.img = read_image('app/assets/Images/messi5.jpg')
            self.img_file_name = 'messi5.jpg'
            
        code_placeholder = st.empty()
        st.write("Upload image: ")
        self.uploader()
        
        with code_placeholder:
            st.subheader("Code")
            st.code(f"""
                    import cv2
                    img = cv2.imread('{self.img_file_name}')
                    lower_reso = cv2.pyrDown(img)
                    cv2.imshow('Original Image', img)
                    cv2.imshow('Lower Resolution Image', lower_reso)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    """)
        h, w, _ = self.img.shape
        
        st.subheader("Below is the 4 levels in an image pyramid.")
        titles = [f'Original - ({h} x {w})', ]
        low_images = [self.img]
        for i in range(1,5):
            low_image = low_reso(self.img, i)
            low_images.append(low_image)
            lh, lw, _ = low_image.shape 
            titles.append(f"Level - {i} ({lh} x {lw})")
            
        self.grid(1, 5, titles, images=low_images)
        
        st.markdown("Now you can go down the image pyramid with `cv2.pyrUp()` function.")
        
        st.subheader("Code")
        st.code("higher_reso = cv2.pyrUp(lower_reso)")
        st.subheader("Output")
        self.grid(1, 2, titles=["Lower Resolution", "Lower to Higher Resolution"], images=[low_images[1], high_reso(low_images[1])])
        st.subheader("What if we apply `cv2.pyrUp()` to the original image?")
        high_image= high_reso(self.img)
        hh, ww, _ = high_image.shape
        self.grid(1, 2, titles=[f"Original ({h} x {w})", f"Higher Resolution ({hh} x {ww})"], images=[self.img, high_image])
        
        st.info("""
                ℹ️nfo
                
                > Zoom in to see the difference in photos
                """)
        
        st.markdown("""
                    Remember, higher_reso2 is not equal to higher_reso, because once you decrease the resolution, 
                    you lose the information. Below image is 3 level down the pyramid created from smallest image
                    in previous case. Compare it with original image:
                    """)
        
        st.subheader("Parameter")
        level = st.number_input("Pyramid Level: ",min_value=1, max_value=5) 
        levelwise = low_reso(self.img, level)
        hh, ww, _ = levelwise.shape 
        st.subheader("Output")
        self.grid(1, 2, titles=[f"Original ({h} x {w})", f"Lower Resolution ({hh} x {ww}) - Level {level}"], images=[self.img, levelwise])
        
        passage = st.empty()
        
        titles = []
        laplacian_images = []
        level = st.number_input("Laplacian Level: ",min_value=1, value=3, max_value=5) 
        passage.markdown(f"""
                    Laplacian Pyramids are formed from the Gaussian Pyramids. There is no exclusive function for that. 
                    Laplacian pyramid images are like edge images only. Most of its elements are zeros. 
                    They are used in image compression. A level in Laplacian Pyramid is formed by the difference between 
                    that level in Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid. The {level}
                    levels of a Laplacian level will look like below (contrast is adjusted to enhance the contents):
                    """)
        for i in range(1,level+1):
            laplacian_image = laplacian_levels(self.img, i)
            laplacian_images.append(laplacian_image)
            lh, lw = laplacian_image.shape 
            titles.append(f"Laplacian Level - {i} ({lh} x {lw})")
        st.subheader("Output")
        self.grid(1, level, titles, images=laplacian_images,clamp=True)
        
    def ImageBlending(self):
        st.subheader("Image Blending using Pyramids")
        st.markdown("""
                    One application of Pyramids is Image Blending. For example, in image stitching, you will need to stack 
                    two images together, but it may not look good due to discontinuities between images. In that case, image
                    blending with Pyramids gives you seamless blending without leaving much data in the images. One classical
                    example of this is the blending of two fruits, Orange and Apple. See the result now itself to understand 
                    what I am saying:
                    """)
        # reading images
        img1 = read_image('app/assets/Images/apple.jpg')
        img2 = read_image('app/assets/Images/orange.jpg')
        
        pyr_blend, dir_blend = image_blending(img1, img2)
        
        self.grid(2,2, titles=['Apple', 'Orange', 'Pyramid Blending', 'Direct Blending'], images=[img1, img2, pyr_blend, dir_blend])
        
        st.markdown("""
                    Please check the first reference in additional resources; it has full diagrammatic details on image blending, 
                    Laplacian Pyramids, etc. Simply, it is done as follows:

                    1. Load the two images of apple and orange.
                    2. Find the Gaussian Pyramids for apple and orange (in this particular example, the number of levels is 6).
                    3. From Gaussian Pyramids, find their Laplacian Pyramids.
                    4. Now, join the left half of the apple and the right half of the orange in each level of Laplacian Pyramids.
                    5. Finally, from this joint image pyramids, reconstruct the original image.

                    Below is the full code. (For the sake of simplicity, each step is done separately, which may take more memory. 
                    You can optimize it if you want to).

                    """)
        
        
        
        st.subheader("Code")
        st.code("""
                import cv2
                import numpy as np,sys

                A = cv2.imread('apple.jpg')
                B = cv2.imread('orange.jpg')

                # generate Gaussian pyramid for A
                G = A.copy()
                gpA = [G]
                for i in range(6):
                    G = cv2.pyrDown(G)
                    gpA.append(G)

                # generate Gaussian pyramid for B
                G = B.copy()
                gpB = [G]
                for i in range(6):
                    G = cv2.pyrDown(G)
                    gpB.append(G)

                # generate Laplacian Pyramid for A
                lpA = [gpA[5]]
                for i in range(5, 0, -1):
                    GE = cv2.pyrUp(gpA[i])
                    rows, cols, dpt = gpA[i-1].shape
                    GE = cv2.resize(GE, (cols, rows))
                    L = cv2.subtract(gpA[i-1], GE)
                    lpA.append(L)

                # generate Laplacian Pyramid for B
                lpB = [gpB[5]]
                for i in range(5, 0, -1):
                    GE = cv2.pyrUp(gpB[i])
                    rows, cols, dpt = gpB[i-1].shape
                    GE = cv2.resize(GE, (cols, rows))
                    L = cv2.subtract(gpB[i-1], GE)
                    lpB.append(L)

                # Now add left and right halves of images in each level
                LS = []
                for la, lb in zip(lpA, lpB):
                    rows, cols, dpt = la.shape
                    ls = np.hstack((la[:, :cols//2], lb[:, cols//2:]))
                    LS.append(ls)

                # now reconstruct
                ls_ = LS[0]
                for i in range(1, 6):
                    ls_ = cv2.pyrUp(ls_)
                    rows, cols, dpt = ls_.shape
                    LS[i] = cv2.resize(LS[i], (cols, rows))
                    ls_ = cv2.add(ls_, LS[i])

                # image with direct connecting each half
                cols = A.shape[1]
                real = np.hstack((A[:, :cols//2], B[:, cols//2:]))

                cv2.imwrite('Pyramid_blending2.jpg',ls_)
                cv2.imwrite('Direct_blending.jpg',real)
                """)
        st.subheader("Output (Creating Orapple)")
        self.grid(1,2, titles=['Pyramid Blending', 'Direct Blending'], images=[pyr_blend, dir_blend])
        
    
class Contours:
    
    class GettingStarted(ImageProcessing):
        def __init__(self):
            super().__init__()
            
        def Introduction(self):
            st.subheader("Goals")
            st.markdown("""
                        - Understand what contours are.
                        - Learn to find contours, draw contours etc
                        - You will see these functions : `cv2.findContours()`, 
                        `cv2.drawContours()`
                        """)
        
        def What_are_Contours(self):
            st.subheader("What are contours?")
            st.markdown("""
                        Contours can be explained simply as a curve joining all the continuous points (along the boundary), 
                        having same color or intensity. The contours are a useful tool for shape analysis and object detection 
                        and recognition.

                        - For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
                        - `findContours()` function modifies the source image. So if you want source image even after finding contours, 
                        already store it to some other variables.
                        - In OpenCV, finding contours is like finding white object from black background. So remember, object to be 
                        found should be white and background should be black.
                        
                        Let’s see how to find contours of a binary image:
                        """)
            
            if not self.img_file:
                self.img = read_image('app/assets/Images/box.png')
                self.img_file_name = 'box.png'
            
            st.subheader("Code")
            code_placeholder= st.empty()
            
            with st.container(border=True):
                st.subheader("Parameters")
                col1, col2= st.columns(2)
                thresh = col1.selectbox("Threshold options: ", index=0, options=get_flags("THRESH_"))
                retr = col2.selectbox("Retrieval Options: ",index=4, options = get_flags("RETR_"))
                chain = col1.selectbox("Contour Approx: ",index=1, options=get_flags("CHAIN_") )
                color= col2.color_picker("Color", value="#00ff00")
                color= tuple(ImageColor.getcolor(f'{color}','RGB')[::-1])
                thickness = col1.number_input("Thickness", min_value = 1, value=3, max_value=5)
                with col2:
                    self.uploader()
            
            code_placeholder.code(f"""
                    import numpy as np
                    import cv2

                    im = cv2.imread('{self.img_file_name}',cv2.IMREAD_COLOR)
                    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                    ret,thresh = cv2.threshold(imgray,127,255,cv2.{thresh})
                    contours, hierarchy = cv2.findContours(thresh,cv2.{retr},cv2.{chain})
                    for data in contours:
                        print "The contours have this data %r" %data
                    cv2.drawContours(im,contours,-1,{color},{thickness})
                    cv2.imshow('output',im)
                    while True:
                        if cv2.waitKey(6) & 0xff == 27:
                            break
                    """)
            try:
                data, contours = get_started_contours(self.img.copy(),thresh, retr, chain)
                st.subheader("Output")
                with st.expander("Reveal print output"):
                    for i in data:
                        st.write(i)
                st.write("Image Output: ")
                self.grid(1, 2, titles=['Original', 'output'], images=[self.img, draw_contours(self.img.copy(), contours, points=-1, color=color, thickness=3)])
                
            except Exception as e:
                st.error(f"Error ⚠️: {e}")
                
            st.markdown("""
                        See, there are three arguments in `cv2.findContours()` function, 
                        first one is source image, second is contour retrieval mode, 
                        third is contour approximation method. And it outputs the contours
                        and hierarchy. `contours` is a Python list of all the contours in 
                        the image. Each individual contour is a Numpy array of (x,y) coordinates
                        of boundary points of the object.
                        And the source image is modified by `cv2.drawContours()` function.
                        """)
            st.info("""
                    :red[ℹ️] Note
                    
                    > We will discuss second and third arguments and about hierarchy in details later.
                    Until then, the values given to them in code sample will work fine for all images.
                    """)
            
        def How_to_Draw(self):
            st.subheader("How to draw the Contours?")
            st.markdown("""
                        To draw the contours, `cv2.drawContours` function is used. It can also be used to draw any shape provided you have its boundary points. 
                        Its first argument is source and destination image, second argument is the contours which should be passed as a Python list, 
                        third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color,
                        thickness etc.
                        
                        To draw all the contours in an image:
                        
                        ```python
                        cv2.drawContours(img, contours, -1, (0,255,0), 3)
                        ```
                        To draw an individual contour, say 4th contour:
                        
                        ```python
                        cv2.drawContours(img, contours, 3, (0,255,0), 3)
                        ```
                        
                        But most of the time, below method will be useful:
                        
                        ```python
                        cnt = contours[4]
                        cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
                        ```
                        
                        """)
            st.info("""
                    :red[ℹ️] Note
                    > Last two methods are same, but when you go forward, you will see last one is more useful.
                    """)
        
        def Contour_Approx_Method(self):
            st.subheader("Contour Approximation Method")
            exp = st.empty()
            if not self.img_file:
                self.img = read_image('app/assets/Images/square.png')
                self.img_file_name = 'box.png'
                
            data1, contours1 = get_started_contours(self.img.copy(), chain='CHAIN_APPROX_NONE')
            data2, contours2 = get_started_contours(self.img.copy(), chain='CHAIN_APPROX_SIMPLE')
            points1=len([c for c in contours1][0]) if contours1 else 0
            points2=len([c for c in contours2][0]) if contours2 else 0
            exp.markdown(f"""
                        This is the third argument in `cv2.findContours` function. What does it denote actually?

                        Above, we told that contours are the boundaries of a shape with same intensity. 
                        It stores the `(x,y)` coordinates of the boundary of a shape. But does it store all the coordinates ? 
                        That is specified by this contour approximation method.

                        If you pass `cv2.CHAIN_APPROX_NONE`, all the boundary points are stored. But actually do we need all the points? 
                        For eg, you found the contour of a straight line. Do you need all the points on the line to represent that line?
                        No, we need just two end points of that line. This is what `cv2.CHAIN_APPROX_SIMPLE` does. It removes all redundant
                        points and compresses the contour, thereby saving memory.

                        Below image of a rectangle demonstrate this technique. Just draw a circle on all the coordinates in the contour 
                        array (drawn in blue color). 
                        - First image shows points I got with `cv2.CHAIN_APPROX_NONE` ({points1} points) and 
                        - second image shows the one with `cv2.CHAIN_APPROX_SIMPLE` (only {points2} points). 
                        See, how much memory it saves!!!
                    """)
            
            self.grid(1, 2, titles=[f"CHAIN_APPROX_NONE {points1}", f"CHAIN_APPROX_SIMPLE {points2}"], 
                      images=[draw_contours(self.img.copy(), contours1[0][:points1], -1, color=(255, 0, 0)), 
                              draw_contours(self.img.copy(), contours2[0][:points2], -1, color=(255, 0, 0))])
    
    class Features(ImageProcessing):
        def __init__(self):
            super().__init__()

        def Introduction(self):
            st.subheader("Goals")
            st.markdown("""
                        Goal
                        In this article, we will learn

                        - To find the different features of contours, 
                        like area, perimeter, centroid, bounding box etc
                        - You will see plenty of functions related to contours.
                        """)
        
        def Moments(self):
            st.subheader("1. Moments")
            st.markdown("""
                        Image moments help you to calculate some features like center of mass of the object,
                        area of the object etc. Check out the wikipedia page on Image Moments

                        The function `cv.moments()` gives a dictionary of all moment values calculated.
                        See below:
                        """)
            
            if not self.img_file:
                self.img = read_image('app/assets/Images/star.png')
                self.img_file_name = 'star.png'
            
            code_placeholder=st.empty()
            with st.container(border=True):
                st.subheader("Try it yourself:")
                self.uploader()
                st.image(self.img, caption=self.img_file_name, use_column_width=True)
                
            code_placeholder.code(f"""
                    import numpy as np
                    import cv2 as cv
                    
                    img = cv.imread('{self.img_file_name}', cv.IMREAD_GRAYSCALE)
                    assert img is not None, "file could not be read, check with os.path.exists()"
                    ret,thresh = cv.threshold(img, 127, 255, 0)
                    contours,hierarchy = cv.findContours(thresh, 1, 2)
                    
                    cnt = contours[0]
                    M = cv.moments(cnt)
                    print( M )
                    """)
            
            st.subheader("Output")
            with st.expander("Reveal output:"):
                moments = get_moments(self.img)
                st.write(moments)
                
            st.markdown("""
                        From this moments, you can extract useful data like area, centroid etc. 
                        Centroid is given by the relations, 
                        $$C_x = \\frac{M_{10}}{M_{00}}$$
                        and 
                        $$C_y = \\frac{M_{01}}{M_{00}}$$
                        . This can be done as follows:
                        """)
            
            cx, cy = get_centroid(moments)
            st.code(f"""
                    cx = int(M['m10']/M['m00'])
                    >> {cx}
                    cy = int(M['m01']/M['m00'])
                    >> {cy}
                    """)
        
        def Contour_Area(self):
            st.subheader("2. Contour Area")
            st.markdown("""
                        Contour area is given by the function `cv2.contourArea()` or from moments, `M[‘m00’]`.
                        """)
            st.code("""
                    area = cv2.contourArea(cnt)
                    """)
        
        def Contour_Perimeter(self):
            st.subheader("3. Contour Perimeter")
            st.markdown("""
                        It is also called arc length. It can be found out using `cv2.arcLength()` function. 
                        Second argument specify whether shape is a closed contour (if passed True), or just a curve.
                        """)
            st.code("perimeter = cv2.arcLength(cnt,True)")
            
        def Contour_Approximation(self):
            
            st.subheader("4. Contour Approximation")
            
            st.markdown("""
                        It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify.
                        It is an implementation of [Douglas-Peucker algorithm](http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm). 
                        Check the wikipedia page for algorithm and demonstration.

                        To understand this, suppose you are trying to find a square in an image, but due to some problems in the image, 
                        you didn’t get a perfect square, but a “bad shape” (As shown in first image below). Now you can use this function 
                        to approximate the shape. In this, second argument is called epsilon, which is maximum distance from contour to 
                        approximated contour. It is an accuracy parameter. A wise selection of epsilon is needed to get the correct output.
                        """)
            
            st.code("""
                    epsilon = 0.1*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    """)
            if not self.img_file: 
                self.img = read_image("app/assets/Images/approx.jpg")
                
            with st.container(border=True):
                st.subheader("Try it yourself:")
                self.uploader()
                
            with st.expander("Reveal output"):
                st.write(get_contour_approx(self.img))
            
            st.markdown("""
                        Below, in second image, green line shows the approximated curve for epsilon = 10% of arc length. Third image shows 
                        the same for epsilon = 1% of the arc length. Third argument specifies whether curve is closed or not.
                        """)
            st.image(self.img, caption="Contour Approximation", channels="BGR", use_column_width=True)
            
            
        def Convex_Hull(self):
            st.subheader("5. Convex Hul")
            st.markdown("""
                        Convex Hull will look similar to contour approximation, but it is not (Both may provide same results in some cases). 
                        Here, `cv2.convexHull()` function checks a curve for convexity defects and corrects it. Generally speaking, convex 
                        curves are the curves which are always bulged out, or at-least flat. And if it is bulged inside, it is called 
                        convexity defects. For example, check the below image of hand. Red line shows the convex hull of hand. The double-sided
                        arrow marks shows the convexity defects, which are the local maximum deviations of hull from contours.
                        """)
            with st.container(border=True):
                st.subheader("Try it yourself:")
                self.uploader()
                im = st.empty()
                msg = st.empty()
                if not self.img_file: 
                    self.img = read_image("app/assets/Images/convexitydefects.jpg")
                    msg.info("Example Image")
                else:
                    self.img = get_cvx_hull(self.img)
                    msg.success("Uploaded Image")
                try:
                    im.image(self.img, caption="Convex Hul", width=500, channels="BGR", clamp=True)
                except Exception as e:
                    st.error(e)
                
            st.write("There is a little bit things to discuss about it its syntax:")
            st.code("hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]")
            st.markdown("""
                        Arguments details:

                        - `points`: The contours we pass into.
                        - `hull`: The output, normally we avoid it.
                        - `clockwise`: Orientation flag. If it is `True`, the output convex hull is oriented clockwise. 
                        Otherwise, it is oriented counter-clockwise.
                        - `returnPoints`: By default, `True`. Then it returns the coordinates of the hull points. 
                        If `False`, it returns the indices of contour points corresponding to the hull points.

                        So to get a convex hull as in the above image, the following is sufficient:
                        """)
            st.code("hull = cv2.convexHull(cnt)")
            st.markdown("""
                        To understand how to find convexity defects, consider the following explanation:

                        - If you want to find convexity defects, you need to pass `returnPoints = False`.
                        - Let's use the example of a rectangle image.
                        - First, find its contour as `cnt`.
                        - Then, find its convex hull with `returnPoints = True`. You'll get the following values: 
                        `[[[234, 202]], [[51, 202]], [[51, 79]], [[234, 79]]]`, which are the four corner points of the rectangle.
                        - Now, if you do the same with `returnPoints = False`, you'll get the following result: `[[129], [67], [0], [142]]`. 
                        These are the indices of corresponding points in `cnt`. For example, check the first value: `cnt[129] = [[234, 202]]`, 
                        which is the same as the first result (and so on for others).

                        This difference becomes relevant when discussing convexity defects and how to handle them.

                        """)
                
        def Checking_Convexity(self):
            st.subheader("6. Checking Convexity") 
            st.markdown("""
                        There is a function to check if a curve is convex or not, `cv2.isContourConvex()`. 
                        It just return whether True or False. Not a big deal.
                        """)
            st.code("k = cv2.isContourConvex(cnt)")
            with st.container(border=True):
                st.subheader("Try it yourself:")
                self.uploader()
                
                if self.img_file:
                    st.write("**Output**")
                    with st.spinner("Wait.."):
                        time.sleep(1)
                        st.code(get_cvx_hull(self.img, check=True))
            
        def Bounding_Rectangle(self):
            st.subheader("7. Bounding Rectangle")
            st.write("There are two types of bounding rectangles.")
            st.markdown("""
                        #### 7.a. Straight Bounding Rectangle
                        It is a straight rectangle, it doesn’t consider the rotation of the object. 
                        So area of the bounding rectangle won’t be minimum. 
                        It is found by the function `cv2.boundingRect()`.
                        
                        Let `(x,y)` be the top-left coordinate of the rectangle and `(w,h)` be its width and height.
                        """)
            st.code("""
                    x,y,w,h = cv2.boundingRect(cnt)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    """)
            
            st.markdown("""
                        #### 7.b. Rotated Rectangle
                        Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. 
                        The function used is cv2.minAreaRect(). It returns a `Box2D` structure which contains 
                        following detals - `( top-left corner(x,y), (width, height), angle of rotation )`. 
                        But to draw this rectangle, we need `4` corners of the rectangle. It is obtained by the function `cv2.boxPoints()`
                        """)
            st.code("""
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    im = cv2.drawContours(im,[box],0,(0,0,255),2)
                    """)
            
            st.markdown("""
                        Both the rectangles are shown in a single image. Green rectangle shows the normal bounding rect. Red rectangle is the rotated rect.
                        """)
            if not self.img_file: 
                self.img = read_image("app/assets/Images/boundingrect.png")
            st.image(self.img, caption="Bounding Rectangles", channels="BGR", use_column_width=True)
            
        def Minimum_Enclosing_Circle(self):
            st.subheader("8. Minimum Enclosing Circle")
            st.markdown("""
                        Next we find the circumcircle of an object using the function `cv.minEnclosingCircle()`. 
                        It is a circle which completely covers the object with minimum area.
                        """)
            st.code("""
                    (x,y),radius = cv.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv.circle(img,center,radius,(0,255,0),2)
                    """)
            if not self.img_file: 
                self.img = read_image("app/assets/Images/circumcircle.png")
            st.image(self.img, caption="Min Enclosing Circle", channels="BGR", use_column_width=True)
            
        def Fitting_an_Ellipse(self):
            st.subheader("9. Fitting an Ellipse")
            st.markdown("""
                        Next one is to fit an ellipse to an object. It returns the rotated rectangle 
                        in which the ellipse is inscribed.
                        """)
            st.code("""
                    ellipse = cv.fitEllipse(cnt)
                    cv.ellipse(img,ellipse,(0,255,0),2)
                    """)
            if not self.img_file: 
                self.img = read_image("app/assets/Images/fitellipse.png")
            st.image(self.img, caption="Fitting an Ellipse", channels="BGR", use_column_width=True)
        
        def Fitting_a_Line(self):
            st.subheader("10. Fitting a Line")
            st.markdown("""
                        Similarly we can fit a line to a set of points. Below image contains a set of 
                        white points. We can approximate a straight line to it.
                        """)
            st.code("""
                    rows,cols = img.shape[:2]
                    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
                    lefty = int((-x*vy/vx) + y)
                    righty = int(((cols-x)*vy/vx)+y)
                    cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
                    """)
            if not self.img_file: 
                self.img = read_image("app/assets/Images/fitline.jpg")
            st.image(self.img, caption="Fitting a Line", channels="BGR", use_column_width=True)

class Histograms(ImageProcessing):
    def __init__(self):
        pass

class ImageTransformations(ImageProcessing):
    def __init__(self):
        pass

class TemplateMatching(ImageProcessing):
    def __init__(self):
        pass

class HoughLineTransform(ImageProcessing):
    def __init__(self):
        pass

class HoughCircleTransform(ImageProcessing):
    def __init__(self):
        pass

class ImageSegmentation(ImageProcessing):
    def __init__(self):
        pass

class InteractiveForegroundExtraction(ImageProcessing):
    def __init__(self):
        pass