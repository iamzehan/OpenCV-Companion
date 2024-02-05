import cv2
import numpy as np

def bytes_to_image(byte_string):
    # Convert the byte string to a NumPy array
    image_array = np.frombuffer(byte_string, dtype=np.uint8)
    # Decode the NumPy array to an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image