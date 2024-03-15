import streamlit as st 
from utils.gui import footer, menu, images

def main():
    smoothing_images = images.SmoothingImages()
    st.title("Smoothing Images")
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Options: ", 
                                options = [
                                    "Introduction",
                                    "2D Convolution",
                                    "Image Blurring"
                                        ],
                            horizontal=True,
                            label_visibility="collapsed")
    
    if options == "Introduction":
        st.subheader("Goals")
        st.markdown("""
                    Learn to:

                    - Blur images with various low pass filters
                    - Apply custom-made filters to images (2D convolution)
                    """)
    if options == "2D Convolution":
        st.subheader("2D Convolution ( Image Filtering )")
        smoothing_images.uploader()
        smoothing_images.Convolution2D()
    
    if options == "Image Blurring":
        st.subheader("Image Blurring (Image Smoothing)")
        smoothing_images.uploader()
        smoothing_images.ImageBlurring()
    
if __name__ == '__main__':
    st.set_page_config("Smoothing Images", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()