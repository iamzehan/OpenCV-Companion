import streamlit as st
from utils.gui.mouse_events import \
    (Simple_Demo,
    Advanced_Demo
    )
from utils.gui.menu import menu
from utils.gui.footer import footer

# Streamlit app
def main():
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Mouse Events")
    menu()
    st.title("OpenCV Mouse Events 🖱️")

    st.markdown("## Goal")
    st.write("Learn to handle mouse events in OpenCV using `cv.setMouseCallback()`.")
    with st.sidebar.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select:", options=["Simple Demo", "Advanced Demo"])
    
    if options == "Simple Demo":
        Simple_Demo()
    if options == "Advanced Demo":
        Advanced_Demo()
    
if __name__ == "__main__":
    main()
    footer()
