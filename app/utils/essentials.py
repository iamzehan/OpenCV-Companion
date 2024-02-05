import cv2 as cv
import numpy as np
import streamlit as st
def bytes_to_image(byte_string):
    # Convert the byte string to a NumPy array
    image_array = np.frombuffer(byte_string, dtype=np.uint8)
    # Decode the NumPy array to an image using OpenCV
    image = cv.imdecode(image_array, cv.IMREAD_COLOR)
    return image

def video_capture():
    start = st.button("Run it :green[▶️]")
    if start:
        cap = cv.VideoCapture(0)
        video_placeholder = st.empty()
        stop = st.button("Stop :red[🟥]")
        if not cap.isOpened():
            st.error("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Display the resulting frame
            video_placeholder.image(gray, use_column_width=True, output_format="JPEG")
            if stop:
                break
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()