import streamlit as st 
from utils.videos import (
    video_capture,
    process_uploaded_video
)
st.set_page_config(page_icon="app\\assets\\OpenCV_Logo_with_text.png", page_title="Getting Started with Videos")

st.markdown("""
            # Getting Started with Videos 📽️
            ## Goals
            * Learn to read video, display video, and save video.
            * Learn to capture video from a camera and display it.
            * You will learn these functions : `cv.VideoCapture()`, `cv.VideoWriter()`
            """,
            unsafe_allow_html=True)

st.subheader("Capture Video from Camera with `cv.VideoCapture()`")
col1, col2,_, _ = st.columns([1,1,1,1])
start = col1.button("Run :green[▶️]", key="Run 1")
if start: 
    video_capture(col2)
    
else:
    st.code("""
    import numpy as np
    import cv2 as cv
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
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
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    """, line_numbers=True)

with st.expander("Details", expanded=False):
    st.markdown("""
`cap.read()` returns a bool (`True/False`). If the frame is read correctly,
it will be `True`. So you can check for the end of the video by checking this returned value.
Sometimes, cap may not have initialized the capture. In that case, this code shows an error. 
You can check whether it is initialized or not by the method `cap.isOpened()`. 
If it is `True`, OK. Otherwise open it using `cap.open()`.

You can also access some of the features of this video using `cap.get(propId)` method where 
`propId` is a number from `0` to `18`. Each number denotes a property of the video 
(if it is applicable to that video). 
Full details can be seen here: 
[`cv::VideoCapture::get()`](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939). 
Some of these values can be modified using `cap.set(propId, value)`. 
Value is the new value you want.

For example, I can check the frame width and height by `cap.get(cv.CAP_PROP_FRAME_WIDTH)` 
and `cap.get(cv.CAP_PROP_FRAME_HEIGHT)`. It gives me `640x480` by default. 
But I want to modify it to 320x240. 
Just use `ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,320)` and `ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)`.

**Note**
> If you are getting an error, make sure your camera is working fine using any other camera application (like Cheese in Linux).
""")

st.subheader("Playing Video from file")
st.markdown("""Playing video from file is the same as capturing it from camera,
            just change the camera index to a video file name. 
            Also while displaying the frame, use appropriate time for cv.waitKey().
            If it is too less, video will be very fast and if it is too high, 
            video will be slow (Well, that is how you can display videos in slow motion). 
            25 milliseconds will be OK in normal cases.""")

video_file_name='vtest.avi'

video_file = st.file_uploader("Upload a video to see how it works", type=["mp4", "avi", "mov"])
if video_file:
    video_file_name = video_file.name
    col1, col2,_,_ = st.columns([1,1,1,1])
    start = col1.button("Run :green[▶️]", key="Run 2")
    if start and video_file:
        process_uploaded_video(video_file, col2)
    else:
        st.code(f"""
    import numpy as np
    import cv2 as cv
    cap = cv.VideoCapture('{video_file_name}')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    """, line_numbers=True)
else:
    st.code(f"""
    import numpy as np
    import cv2 as cv
    cap = cv.VideoCapture('{video_file_name}')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    """, line_numbers=True)

st.markdown("""
**Note**
> Make sure a proper version of `ffmpeg` or `gstreamer` is installed. 
Sometimes it is a headache to work with video capture, mostly due to wrong installation of 
`ffmpeg/gstreamer`.
""")

