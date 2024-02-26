import cv2 as cv
import numpy as np

def read_image(path):
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

