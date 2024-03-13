import cv2 as cv
import numpy as np

def blank_image(height, width, channel):
    img = np.zeros((height, width, channel), np.uint8)
    return img

def read_image(path, grey=False):
    if grey:
        img = cv.imread(path, 0)
    else:
        img = cv.imread(path)
    return img

def get_shape(img):
    return img.shape

def get_size(img):
    return img.size

def get_dtype(img):
    return img.dtype

def split_channels(img):
    b, g, r = cv.split(img)
    return b, g, r

def merge_channels(b, g, r):
    img = cv.merge((b,g,r))
    return img

def bytes_to_image(byte_string):
    # Convert the byte string to a NumPy array
    image_array = np.frombuffer(byte_string, dtype=np.uint8)
    # Decode the NumPy array to an image using OpenCV
    image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    return image

def load_by_pixels(img, dimensions:list=[100,100], color=None)->list:
    px = img[dimensions[0]-1, dimensions[1]-1, color]
    return px

def list_to_np_array(lst):
    return np.array(lst)

def make_borders(img1):
    
    BLUE = [255,0,0]
    replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
    constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
    
    return replicate, reflect, reflect101, wrap, constant

def resize(img, h, w):
    return cv.resize(img, (h, w))

def add_two_img(img1, img2, alpha=0.7, beta=0.3, gamma=0, blend=False):
    h1, w1, _ = get_shape(img1)
    h2, w2, _ = get_shape(img2)
    h, w = max(h1, h2), max(w1, w2)
    img1, img2 = resize(img1, h, w), resize(img2, h, w)
    if blend:
        return cv.addWeighted(img1, alpha, img2, beta, gamma)
    else:
        return cv.add(img1,img2) 

def bitwise_ops(img1, img2):
    h, w, c = img1.shape
    h, w = h//5, w//5
    img2 = resize(img2, h, w)
    rows,cols, channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2, img2, mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst
    
    return img1

def performance_measure(img):
    e1 = cv.getTickCount()
    for i in range(5,49,2):
        img = cv.medianBlur(img,i)
        e2 = cv.getTickCount()
        t = (e2 - e1)/cv.getTickFrequency()
    return t

def colorspace_flags():
    flags = [i for i in dir(cv) if i.startswith('COLOR_')]
    return flags

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

def find_hsv_values(color):
    color = np.uint8([[color]])
    hsv = cv.cvtColor(color,cv.COLOR_BGR2HSV)
    return hsv

def scaling(img, fx=2, fy=2, inter="INTER_CUBIC"):
    
    interpolations = {
                    "INTER_CUBIC": cv.INTER_CUBIC,
                    "INTER_AREA" :cv.INTER_AREA,
                    "INTER_LINEAR": cv.INTER_LINEAR
                      }
    
    res = cv.resize(img, None, fx=fx, fy=fy, interpolation = interpolations[inter])
    return res

def translation(img, shift:int):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,shift]])
    dst = cv.warpAffine(img,M,(cols,rows))
    return img, dst

def rotation(img, rotaion):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows,cols= img.shape

    M = cv.getRotationMatrix2D((cols/2,rows/2), rotaion, 1)
    dst = cv.warpAffine(img,M,(cols,rows))
    return dst

def affine_transform(img):
    rows,cols,ch = img.shape

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv.getAffineTransform(pts1,pts2)

    dst = cv.warpAffine(img,M,(cols,rows))
    
    return dst

def perspective_transform(img, pts1=[[56,65],[368,52],[28,387],[389,390]]):
    rows,cols,ch = img.shape
    img = resize(img, 300, 300)
    pts1 = np.float32(pts1)
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv.getPerspectiveTransform(pts1,pts2)

    dst = cv.warpPerspective(img,M,(300,300))
    return dst

def simple_thresholding(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret,thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    return images

def adaptive_thresholding(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    
    images = [img, th1, th2, th3]
    
    return images

def otsus_binarization(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
    return images

def conv2D(img, dim=(5,5), div=25):
    kernel = np.ones(dim,np.float32)/div
    dst = cv.filter2D(img,-1,kernel)
    return dst

def averaging(img, dim = (5,5)):
    return cv.blur(img, dim)

def gaussian_blur(img, dim=(5,5), intensity=0):
    return cv.GaussianBlur(img, dim, intensity)

def median_blur(img, intensity=5):
    return cv.medianBlur(img,intensity)

def bilateral_filter(img, d, sigma_color, sigma_space):
    return cv.bilateralFilter(img, d, sigma_color, sigma_space)

def get_binary_image(img):
    _, binary_image = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return binary_image

def erosion(img, dim=(5,5), iterations=1):
    img = get_binary_image(img)
    kernel = np.ones(dim,np.uint8)
    erosion = cv.erode(img, kernel, iterations = iterations)
    return erosion

def dilation(img, dim=(5,5), iterations=1):
    erode = erosion(img, dim, iterations)
    kernel = np.ones(dim, np.uint8)
    dilation = cv.dilate(img, kernel, iterations=iterations)
    return erode, dilation

def opening(img, dim=(5,5)):
    img = get_binary_image(img)
    kernel = np.ones(dim, np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return opening

def closing(img, dim=(5,5)):
    img = get_binary_image(img)
    kernel = np.ones(dim, np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return closing

def morph_gradient(img, dim=(5,5)):
    img = get_binary_image(img)
    kernel = np.ones(dim, np.uint8)
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    return gradient 

def top_hat(img, dim=(9,9)):
    img = get_binary_image(img)
    kernel = np.ones(dim, np.uint8)
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    return tophat

def black_hat(img, dim=(9,9)):
    img = get_binary_image(img)
    kernel = np.ones(dim, np.uint8)
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    return blackhat

def get_morph():
    flags = [i for i in dir(cv) if i.startswith('MORPH_')]
    return flags

def get_structuring_element(key, dim):
    morph_operations = {
    "MORPH_BLACKHAT": cv.MORPH_BLACKHAT,
    "MORPH_CLOSE": cv.MORPH_CLOSE,
    "MORPH_CROSS": cv.MORPH_CROSS,
    "MORPH_DILATE": cv.MORPH_DILATE,
    "MORPH_ELLIPSE": cv.MORPH_ELLIPSE,
    "MORPH_ERODE": cv.MORPH_ERODE,
    "MORPH_GRADIENT": cv.MORPH_GRADIENT,
    "MORPH_HITMISS": cv.MORPH_HITMISS,
    "MORPH_OPEN": cv.MORPH_OPEN,
    "MORPH_RECT": cv.MORPH_RECT,
    "MORPH_TOPHAT": cv.MORPH_TOPHAT,
    }
    
    return cv.getStructuringElement(morph_operations[key],dim)

def img_gradient(img, options = None):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    if options == "Theory":
        laplacian = cv.Laplacian(img,cv.CV_64F)
        sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
        sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
        return [img, laplacian, sobelx, sobely]
    if options == "One important matter!":
        # Output dtype = cv2.CV_8U
        sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_8u = np.uint8(abs_sobel64f)
        return [img, sobelx8u, sobel_8u]

def Canny(img, minVal = 100, maxVal=200):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img,minVal,maxVal)
    return img, edges

def low_reso(img, level=1):
    i = 1
    while i<=level:
        img = cv.pyrDown(img)
        i+=1
    return img

def high_reso(img):
    return cv.pyrUp(img)