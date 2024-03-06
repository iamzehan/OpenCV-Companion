import streamlit as st 
from utils.gui.footer import footer
from utils.gui.menu import menu

def main():
    st.title("Core Operations in OpenCV")

    with st.container(border=True):
        st.page_link("pages/7_Basic_Operations_on_Images.py", label="👉 **Basic Operations on Images**", use_container_width=True)
        col1, col2, col3 = st.columns([2,8,2])
        col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/pixel_ops.jpg")
        col2.markdown("Learn to read and edit pixel values, working with image ROI and other basic operations.")
        col3.page_link("pages/7_Basic_Operations_on_Images.py", label="**Learn**➡️", use_container_width=True)

    with st.container(border=True):
        st.page_link("pages/8_Arithmetic_Operations_on_Images.py", label="👉 **Arithmetic Operations on Images**")
        col1, col2, col3 = st.columns([2,8,2])
        col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/image_arithmetic.jpg")
        col2.markdown("Perform arithmetic operations on images")
        col3.page_link("pages/8_Arithmetic_Operations_on_Images.py", label="**Learn** ➡️")

    with st.container(border=True):
        st.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label=" 👉 **Performance Measurement and Improvement Techniques**")
        col1, col2, col3 = st.columns([2,8,2])
        col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/speed.jpg")
        col2.markdown("Getting a solution is important. But getting it in the fastest way is more important. Learn to check the speed of your code, optimize the code etc.")
        col3.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label="**Learn** ➡️")
        
    with st.container(border=True):
        st.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="👉 **Mathematical Tools in OpenCV**", use_container_width=True)
        col1, col2, col3 = st.columns([2,8,2])
        col1.image("https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/maths_tools.jpg")
        col2.markdown("Learn some of the mathematical tools provided by OpenCV like PCA, SVD etc.")
        col3.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="**Learn ➡️**")
        
    col1, col2, col3 = st.columns([2,8,2])
    col3.page_link("pages/7_Basic_Operations_on_Images.py", label="Next ➡️", use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                   page_title="Core Operations in OpenCV")
    menu()
    main()
    footer()