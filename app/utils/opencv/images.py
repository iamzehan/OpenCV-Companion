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

def add_two_img(img1, img2):
    h1, w1, _ = get_shape(img1)
    h2, w2, _ = get_shape(img2)
    h, w = max(h1, h2), max(w1, w2)
    img1, img2 = resize(img1, h, w), resize(img2, h, w)
    return cv.add(img1,img2) 