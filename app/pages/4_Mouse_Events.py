import streamlit as st
from utils.gui import menu, footer, mouse_events

# Streamlit app
def main():
    me = mouse_events.MouseEvents()
    st.title("OpenCV Mouse Events 🖱️")
    st.markdown("## Goal")
    st.write("Learn to handle mouse events in OpenCV using `cv.setMouseCallback()`.")
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select:", options=["Simple Demo", "Advanced Demo"], horizontal=True, label_visibility='collapsed')
    
    if options == "Simple Demo":
        me.Simple_Demo()
    if options == "Advanced Demo":
        me.Advanced_Demo()
    
if __name__ == "__main__":
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Mouse Events")
    menu.menu()
    main()
    footer.footer()
