import streamlit as st
from utils.gui.mouse_events import Simple_Demo

# Streamlit app
def main():
    st.title("OpenCV Mouse Events")

    st.markdown("## Goal")
    st.write("Learn to handle mouse events in OpenCV using `cv.setMouseCallback()`.")

    options = st.sidebar.selectbox("Select:", options=["Simple Demo", "Advanced Demo"])
    
    if options == "Simple Demo":
        Simple_Demo()
    if options == "Advanced Demo":
        pass
if __name__ == "__main__":
    main()
