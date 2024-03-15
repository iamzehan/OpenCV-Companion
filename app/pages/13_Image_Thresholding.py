import streamlit as st 
from utils.gui import footer, menu
from utils.gui.images import ImageThresholding

def main():
    image_thresholding = ImageThresholding()

    # Streamlit app
    st.title("Image Thresholding")

    # Select box for choosing the thresholding method
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select Thresholding Method", 
                                        ["Introduction",
                                         "Simple Thresholding", 
                                        "Adaptive Thresholding", 
                                        "Otsu's Binarization"],
                                        horizontal=True,
                                        label_visibility="collapsed")

    if options == "Introduction":
        st.subheader("Goals")
        st.markdown("""
                    - In this tutorial, you will learn Simple thresholding, Adaptive thresholding, 
                    Otsu’s thresholding etc.
                    - You will learn these functions : `cv2.threshold`, `cv2.adaptiveThreshold` etc.
                    """)
    if options == "Simple Thresholding":
        st.subheader("Simple Thresholding")
        image_thresholding.uploader()
        image_thresholding.Simple_Thresholding()
        
    elif options == "Adaptive Thresholding":
        st.subheader("Adaptive Thresholding")
        image_thresholding.uploader()
        image_thresholding.Adaptive_Thresholding()
        
    elif options == "Otsu's Binarization":
        st.subheader("Otsu's Binarization")
        image_thresholding.uploader()
        image_thresholding.Otsus_Binarization()

if __name__ == '__main__':
    st.set_page_config("Image Thresholding", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()