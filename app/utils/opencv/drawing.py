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

def draw_circle(center=(447, 63), radius=63, color=(0, 255, 0), thickness=-1):
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    cv.circle(img,center, radius, color, thickness)
    return img

def draw_ellipse(center_coordinates = (256,256),
                 axesLength = (100, 50), 
                 angle = 0, 
                 startAngle=0, 
                 endAngle=180, 
                 color=(255,0,0),
                 thickness=-1):
    img = np.zeros((512,512,3), np.uint8)
    cv.ellipse(img, center_coordinates, axesLength, 
           angle, startAngle, endAngle, color, thickness) 
    return img

def draw_polygon(pts=[[10,5],[20,30],[70,20],[50,10]],
                 join=True, color=(0, 255,255)):
    
    img = np.zeros((512,512,3), np.uint8)
    pts =  np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], join, color)
    return img

def draw_text(text='OpenCV',
              position=(10, 500),
              font='HERSHEY_SIMPLEX',
              font_scale=4,
              color = (255,255,255),
              thickness=2,
              lineType='LINE_AA'):
    
    font_dict = {
    'HERSHEY_SIMPLEX': cv.FONT_HERSHEY_SIMPLEX,
    'HERSHEY_PLAIN': cv.FONT_HERSHEY_PLAIN,
    'HERSHEY_DUPLEX': cv.FONT_HERSHEY_DUPLEX,
    'HERSHEY_COMPLEX': cv.FONT_HERSHEY_COMPLEX,
    'HERSHEY_TRIPLEX': cv.FONT_HERSHEY_TRIPLEX,
    'HERSHEY_COMPLEX_SMALL': cv.FONT_HERSHEY_COMPLEX_SMALL,
    'HERSHEY_SCRIPT_SIMPLEX': cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    'HERSHEY_SCRIPT_COMPLEX': cv.FONT_HERSHEY_SCRIPT_COMPLEX,
}
    line_type_dict = {
    'LINE_AA': cv.LINE_AA,
    'LINE_4': cv.LINE_4,
    'LINE_8': cv.LINE_8,
    'LINE_AA_8': cv.LINE_AA | cv.LINE_8,
}
    font = font_dict[font]
    lineType=line_type_dict[lineType]
    img = np.zeros((512,512,3), np.uint8)
    cv.putText(img, text, position, font,
               font_scale, color, thickness, lineType)
    return img

