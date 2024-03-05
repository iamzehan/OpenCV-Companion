import streamlit as st 
from utils.gui import footer, menu
from utils.gui.images import ImageThresholding

def main():
    image_thresholding = ImageThresholding()

    # Streamlit app
    st.title("Image Thresholding")

    # Select box for choosing the thresholding method
    with st.sidebar.container(border=True):
        st.subheader("Topics")
        selected_option = st.radio("Select Thresholding Method", 
                                        ["Simple Thresholding", 
                                            "Adaptive Thresholding", 
                                            "Otsu's Binarization"],
                                        label_visibility="collapsed")

    # Call the corresponding function based on the selected option
    if selected_option == "Simple Thresholding":
        st.subheader("Simple Thresholding")
        image_thresholding.Simple_Thresholding()
        
    elif selected_option == "Adaptive Thresholding":
        st.subheader("Adaptive Thresholding")
        image_thresholding.Adaptive_Thresholding()
        
    elif selected_option == "Otsu's Binarization":
        st.subheader("Otsu's Binarization")
        image_thresholding.Otsus_Binarization()

if __name__ == '__main__':
    st.set_page_config("Image Thresholding", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()