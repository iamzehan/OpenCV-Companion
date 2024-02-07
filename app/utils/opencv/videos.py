import tempfile
import cv2 as cv
import streamlit as st

def video_capture(col2=st, path=None):
    cap = cv.VideoCapture(path) if path else cv.VideoCapture(0) 
    video_placeholder = st.empty()
    stop = col2.button("Stop :red[🟥]")
    if not cap.isOpened():
        st.error("Cannot open camera")
        exit()
    else:
        st.success("Reading from Camera Feed")
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
    
def process_uploaded_video(video_file, col2=st):
    # Convert the video content to a NumPy array
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    # Create a VideoCapture object using the NumPy array as input
    cap = cv.VideoCapture(tfile.name)
    stop = col2.button("Stop :red[🟥]")
    video_placeholder = st.empty()
    # Check if the VideoCapture object was created successfully
    if not cap.isOpened():
        st.error("Error opening video file.")
        return
    else:
        st.success("Reading from your Uploaded File ")
    # Display video frames in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame using Streamlit
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check for user interruption (e.g., closing the browser window)
        if stop:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            break
    
    # Release the VideoCapture object
    cap.release()
    
