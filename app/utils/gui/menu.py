import streamlit as st

def menu():
    
    def gui_features_menu():
        st.page_link("pages/0_GUI_Features_In_OpenCV.py", label = "**GUI Features in OpenCV**")
        with st.expander("More.."):
            _, col2 = st.columns([1,11])
            with col2:    
                st.page_link("pages/1_Getting_Started_with_Images.py", label = "Getting Started with Images")
                st.page_link("pages/2_Getting_Started_with_Videos.py", label = "Getting Started with Videos")
                st.page_link("pages/3_Drawing_Functions_in_OpenCV.py", label = "Drawing Functions")
                st.page_link("pages/4_Mouse_Events.py", label = "Mouse Events")
                st.page_link("pages/5_Trackbar.py", label = "Trackbar")
                
    def core_operations_menu():
        st.page_link("pages/6_Core_Operations.py", label="**Core Operations**")
        with st.expander("More.."):
            _, col2 = st.columns([1,11])
            with col2:    
                st.page_link("pages/7_Basic_Operations_on_Images.py", label="Basic Operations")
                st.page_link("pages/8_Arithmetic_Operations_on_Images.py", label ="Arithmetic Opertations")
                st.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label ="Performance")
                st.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="Math Tools")

    def image_processing_menu():
        st.page_link("pages/11_Image_Processing_in_OpenCV.py", label="**Image Processing**")
        with st.expander("More.."):
            _, col2 = st.columns([1,11])
            with col2:    
                st.page_link("pages/12_Changing_Color_Spaces.py", label = "Changing Colorspaces")
                st.page_link("pages/13_Image_Thresholding.py", label = "Image Thresholding")
                st.page_link("pages/14_Smoothing_Images.py", label = "Smoothing Images")
                st.page_link("pages/15_Geometric_Transformations_of_Images.py", label = "Geometric Transformations of Images")
                st.page_link("pages/16_Morphological_Transformations.py", label = "Morphological Transformations")
                st.page_link("pages/17_Image_Gradients.py", label = "Image Gradients")
                st.page_link("pages/18_Canny_Edge_Detection.py", label = "Canny Edge Detection")
                st.page_link("pages/19_Image_Pyramids.py", label = "Image Pyramids")
                
                st.page_link("pages/20_Contours_in_OpenCV.py", label = "Contours in OpenCV")
                st.page_link("pages/20_1_Contours_GettingStarted.py", label = "Contours: Getting Started")
                st.page_link("pages/20_2_Contour_Features.py", label = "Contour Features")
                st.page_link("pages/20_3_Contour_Properties.py", label = "Contour Properties")
                st.page_link("pages/20_4_Contour_More_Functions.py", label = "Contour: More Functions.py")
                st.page_link("pages/20_5_Contours_Hierarchy.py", label = "Contours Hierarchy")
                    
                st.page_link("pages/21_Histograms_in_OpenCV.py", label = "Histograms in OpenCV")
                st.page_link("pages/22_Image_Transforms_in_OpenCV.py", label = "Image Transforms in OpenCV")
                st.page_link("pages/23_Template_Matching.py", label = "Template Matching")
                st.page_link("pages/24_Hough_Line_Transform.py", label = "Hough Line Transform")
                st.page_link("pages/25_Hough_Circle_Transform.py", label = "Hough Circle Transform")
                st.page_link("pages/26_Image_Segmentation.py", label = "Image Segmentation with Watershed Algorithm")
                st.page_link("pages/27_Interactive_Foreground_Extraction.py", label = "Interactive Foreground Extraction using GrabCut Algorithm")
    
    with st.sidebar:
        st.page_link('Getting_Started_with_OpenCV.py', label="Getting Started")
        gui_features_menu()
        st.divider()
        core_operations_menu()
        st.divider()
        image_processing_menu()
        st.divider()

if __name__ == "__main__":
    menu()