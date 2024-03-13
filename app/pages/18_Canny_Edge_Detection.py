import streamlit as st 
from utils.gui import footer, menu, images

def main():
    canny_edge_detect = images.CannyEdgeDetection()
    st.title("Canny Edge Detection")
    
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Options: ", 
                        options=["Introduction", "Theory", "Canny Edge Detection"],
                        horizontal=True,
                        label_visibility="collapsed")
    
    with st.container(border=True):
        
        if options == "Introduction":
            st.subheader("Goals")
            st.markdown("""
                        In this chapter, we will learn about

                        - Concept of Canny edge detection
                        - OpenCV functions for that : `cv2.Canny()`
                        
                        """)
            
        elif options=="Theory":
            canny_edge_detect.Theory()
        
        elif options == "Canny Edge Detection":
            canny_edge_detect.side_bar()
            canny_edge_detect.Canny_Edge_Detection()

if __name__ == '__main__':
    st.set_page_config("Canny Edge Detection", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()