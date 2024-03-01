import streamlit as st 
import time
from utils.gui.menu import menu
from utils.gui.footer import footer

def main():
    pages = [{
        "link":"pages/12_Changing_Color_Spaces.py",
        "label":"Changing Colorspaces",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/colorspace.jpg",
        "description":"Learn to change images between different color spaces. Plus learn to track a colored object in a video."
    },
             {
        "link":"pages/13_Image_Thresholding.py",
        "label":"Image Thresholding",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/thresh.jpg",
        "description":"Learn to convert images to binary images using global thresholding, Adaptive thresholding, Otsu's binarization etc"
    },
    
                 {
        "link":"pages/14_Smoothing_Images.py",
        "label":"Smoothing Images",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/blurring.jpg",
        "description":"	Learn to blur the images, filter the images with custom kernels etc."
    },
    
    {
        "link":"pages/15_Geometric_Transformations_of_Images.py",
        "label":"Geometric Transformations of Images",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/geometric.jpg",
        "description":"Learn to apply different geometric transformations to images like rotation, translation etc."
    },
    {
        "link":"pages/16_Morphological_Transformations.py",
        "label":"Morphological Transformations",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/morphology.jpg",
        "description":"Learn about morphological transformations like Erosion, Dilation, Opening, Closing etc"
    },
             {
        "link":"pages/17_Image_Gradients.py",
        "label":"Image Gradients",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/gradient.jpg",
        "description":"Learn to find image gradients, edges etc."
    },
             {
        "link":"pages/18_Canny_Edge_Detection.py",
        "label":"Canny Edge Detection",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/canny.jpg",
        "description":"Learn to find edges with Canny Edge Detection"
    },
             {
        "link":"pages/19_Image_Pyramids.py",
        "label":"Image Pyramids",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/pyramid.png",
        "description":"Learn about image pyramids and how to use them for image blending"
    },
             {
        "link":"pages/20_Contours_in_OpenCV.py",
        "label":"Contours in OpenCV",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contours.jpg",
        "description":"All about Contours in OpenCV"
    },
             {
        "link":"pages/21_Histograms_in_OpenCV.py",
        "label":"Histograms in OpenCV",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/histogram.jpg",
        "description":"All about histograms in OpenCV"
    },
             {
        "link":"pages/22_Image_Transforms_in_OpenCV.py",
        "label":"Image Transforms in OpenCV",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/transforms.jpg",
        "description":"Meet different Image Transforms in OpenCV like Fourier Transform, Cosine Transform etc."
    },
             {
        "link":"pages/23_Template_Matching.py",
        "label":"Template Matching",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/template.jpg",
        "description":"Learn to search for an object in an image using Template Matching"
    },
             {
        "link":"pages/24_Hough_Line_Transform.py",
        "label":"Hough Line Transform",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/houghlines.jpg",
        "description":"Learn to detect lines in an image"
    },
             {
        "link":"pages/25_Hough_Circle_Transform.py",
        "label":"Hough Circle Transform",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/houghcircles.jpg",
        "description":"Learn to detect circles in an image"
    },
             {
        "link":"pages/26_Image_Segmentation.py",
        "label":"Image Segmentation with Watershed Algorithm",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/watershed.jpg",
        "description":"Learn to segment images with watershed segmentation"
    },

             {
        "link":"pages/27_Interactive_Foreground_Extraction.py",
        "label":"Interactive Foreground Extraction using GrabCut Algorithm",
        "image":"https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/grabcut1.jpg",
        "description":"Learn to extract foreground with GrabCut algorithm"
    }]

    st.title("Image Processing in OpenCV")
    
    for i in range(len(pages)):
        with st.container(border=True):
            with st.spinner("Please wait.."):
                time.sleep(0.2)
                link, label, image, description= pages[i]['link'], pages[i]['label'], pages[i]['image'], pages[i]['description']
                st.page_link(f"{link}", label=f"{i+1} . **{label}**", use_container_width=True)
                col1, col2, col3 = st.columns([2,8,2])
                col1.image(f"{image}", use_column_width=True)
                col2.markdown(f"{description}")
                col3.page_link(f"{link}", label="**Learn**➡️", use_container_width=True)
                
if __name__ == '__main__':
    st.set_page_config(page_title='Image Processing in OpenCV', page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu()
    main()
    footer()
    