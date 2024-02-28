import streamlit as st

def menu():
    def core_operations_menu():
        st.sidebar.page_link("pages/6_Core_Operations_🔴.py", label="**Core Operations**")
        col1, col2 = st.sidebar.columns([1,11])
        with col2:    
            st.page_link("pages/7_Basic_Operations_on_Images.py", label="Basic Operations")
            st.page_link("pages/8_Arithmetic_Operations_on_Images.py", label ="Arithmetic Opertations")
            st.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label ="Performance")
            st.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="Math Tools")

    def gui_features_menu():
        st.sidebar.page_link("pages/0_GUI_Features_In_OpenCV.py", label = "**GUI Features in OpenCV**")
        col1, col2 = st.sidebar.columns([1,11])
        with col2:    
            st.page_link("pages/1_Getting_Started_with_Images.py", label = "Getting Started with Images")
            st.page_link("pages/2_Getting_Started_with_Videos.py", label = "Getting Started with Videos")
            st.page_link("pages/3_Drawing_Functions_in_OpenCV.py", label = "Drawing Functions")
            st.page_link("pages/4_Mouse_Events.py", label = "Mouse Events")
            st.page_link("pages/5_Trackbar.py", label = "Trackbar")
    
    def image_processing_menu():
        st.sidebar.page_link("pages/11_Image_Processing_in_OpenCV.py", label="Image Processing")
        col1, col2 = st.sidebar.columns([1,11])
        with col2:    
            st.page_link("pages/12_Changing_Color_Spaces.py", label = "Changing Colorspaces")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Image Thresholding")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Smoothing Images")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Geometric Transformations of Images")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Morphological Transformations")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Image Gradients")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Canny Edge Detection")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Image Pyramids")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Contours in OpenCV")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Histograms in OpenCV")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Image Transforms in OpenCV")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Template Matching")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Hough Line Transform")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Hough Circle Transform")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Image Segmentation with Watershed Algorithm")
            st.page_link("pages/11_Image_Processing_in_OpenCV.py", label = "Interactive Foreground Extraction using GrabCut Algorithm")
    
    st.sidebar.page_link('Getting_Started_with_OpenCV.py', label="Getting Started")
    gui_features_menu()
    core_operations_menu()
    image_processing_menu()

