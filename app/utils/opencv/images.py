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