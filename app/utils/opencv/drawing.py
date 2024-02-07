import numpy as np
import cv2 as cv 

def draw_line(start=(0,0), end=(511, 511), color=(255,0, 0), thickness=5):
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    cv.line(img,start, end, color, thickness)
    return img

def draw_rectangle(top_left=(384,0), bottom_right=(510, 128), color=(0,255,0), thickness=5):
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    cv.rectangle(img, top_left, bottom_right, color, thickness)
    cv.circle(img, (top_left), 5, (255,255,240), -1)
    cv.circle(img, (bottom_right), 5, (255,255,255), -1)
    return img

def draw_circle(center=(447, 63), radius=63, color=(0, 0, 255), thickness=-1):
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    cv.circle(img,center, radius, color, thickness)
    return img