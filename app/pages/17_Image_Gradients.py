import streamlit as st 
from utils.gui import footer, menu, images

def main():
    image_gradient = images.ImageGradients()
    st.title("Image Gradients")
    with st.container(border=True):
        options = st.radio("Options: ",
                           options=["Introduction", "Theory", "One important matter!"],
                           horizontal=True,
                           label_visibility="collapsed")
    
    with st.container(border=True):
        if options == "Introduction":
            st.subheader("Goals")
            st.markdown("""
                        In this chapter, we will learn to:

                        - Find Image gradients, edges etc
                        - We will see following functions : `cv2.Sobel()`, `cv2.Scharr()`, `cv2.Laplacian()` etc.
                        """)
        
        elif options == "Theory":
            st.subheader("Theory")
            image_gradient.Theory(options)
        
        elif options == "One important matter!":
            st.subheader("One important matter!")
            image_gradient.Important(options)
        
if __name__ == '__main__':
    st.set_page_config("Image Gradients", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()