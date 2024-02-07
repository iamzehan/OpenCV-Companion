import streamlit as st
from utils.opencv.videos import video_capture, process_uploaded_video

def Capture_Video_from_Webcam():
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



def Play_Video_from_File():
    # Playing Video from file
    st.subheader("Playing Video from file")
    st.markdown("""Playing video from file is the same as capturing it from camera,
                just change the camera index to a video file name. 
                Also while displaying the frame, use appropriate time for cv.waitKey().
                If it is too less, video will be very fast and if it is too high, 
                video will be slow (Well, that is how you can display videos in slow motion). 
                25 milliseconds will be OK in normal cases.""")

    def show_code(video_file_name):
        st.code(f"""
        import numpy as np
        import cv2 as cv
        cap = cv.VideoCapture('<path>/{video_file_name}')
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

    video_file_name='vtest.avi'
    video_file = st.file_uploader("Upload a video to see how it works", type=["mp4", "avi", "mov"])
    if video_file:
        video_file_name = video_file.name
        col1, col2,_, _ = st.columns([1,1,1,1])
        start = col1.button("Run :green[▶️]", key="Run 2")
        if start and video_file:
            process_uploaded_video(video_file, col2)
        else:
            show_code(video_file_name)
            st.success("**Notice:** How the path have changed❗")
    else:
        show_code(video_file_name)
        st.error("Please upload a video to see how it works")

    st.markdown("""
    **Note**
    > Make sure a proper version of `ffmpeg` or `gstreamer` is installed. 
    Sometimes it is a headache to work with video capture, mostly due to wrong installation of 
    `ffmpeg/gstreamer`.
    """)
    
def Save_Video():
    st.markdown("""
                # Video Processing and Saving in OpenCV

When working with videos, capturing and processing frames frame-by-frame is a common task. Unlike images, saving videos involves a bit more effort. In OpenCV, for images, you can use `cv.imwrite()`. However, for videos, you need to create a `VideoWriter` object.

To save a video, follow these steps:

1. Create a `VideoWriter` object.
   - Specify the output file name (e.g., output.avi).
   - Specify the FourCC code (details in the next paragraph).
   - Provide the frames per second (fps) and frame size.
   - Set the isColor flag. If it's True, the encoder expects color frames; otherwise, it works with grayscale frames.

2. FourCC (Four-Character Code):
   - FourCC is a 4-byte code used to specify the video codec.
   - The list of available codes can be found on [fourcc.org](http://www.fourcc.org/).
   - It is platform-dependent.
   - Example codes for different platforms:
     - In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is preferable; MJPG results in high-size video; X264 gives very small size video)
     - In Windows: DIVX (More to be tested and added)
     - In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
   - FourCC code can be passed as `cv.VideoWriter_fourcc('M','J','P','G')` or `cv.VideoWriter_fourcc(*'MJPG')` for MJPG.

3. Example code:
   The following code captures from a camera, flips every frame in the vertical direction, and saves the video.

```python
import cv2 as cv

# Set up video capture
cap = cv.VideoCapture(0)  # 0 represents the default camera

# Define the codec and create a VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Example codec (XVID)
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically
    frame = cv.flip(frame, 1)

    # Write the flipped frame to the output video file
    out.write(frame)

    cv.imshow('Flipped Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv.destroyAllWindows()
""")