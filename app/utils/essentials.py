import cv2 as cv
import numpy as np
import streamlit as st

def bytes_to_image(byte_string):
    # Convert the byte string to a NumPy array
    image_array = np.frombuffer(byte_string, dtype=np.uint8)
    # Decode the NumPy array to an image using OpenCV
    image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    return image

def video_capture(col2):
    cap = cv.VideoCapture(0)
    video_placeholder = st.empty()
    stop = col2.button("Stop :red[🟥]", use_container_width=True)
    if not cap.isOpened():
        st.error("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            st.error("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Display the resulting frame
        video_placeholder.image(gray, use_column_width=True, output_format="JPEG")
        if stop:
            video_placeholder=st.empty()
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()