import streamlit as st 
from utils.gui.footer import footer
from utils.gui.images import ArithmeticOperations

if __name__ == '__main__':
    options = st.sidebar.selectbox(label="Navigate: ",options=["Introduction",
                                                               "Image Addition",
                                                               "Image Blending",
                                                               "Bitwise Operations"],
                                   label_visibility="collapsed")
    arithmeticOps = ArithmeticOperations() 
    st.markdown("# Arithmetic Operations on Images")
    if options == "Introduction":
        st.markdown("""
                    ## Goal
                    - Learn several arithmetic operations on images like 
                    addition, subtraction, bitwise operations etc.
                    - You will learn these functions : 
                    `cv2.add()`, `cv2.addWeighted()` etc.
                    """)
        
    if options == "Image Addition":
        arithmeticOps.side_bar(multiple=True)
        arithmeticOps.Image_Addition()
    if options == "Image Blending":
        pass
    if options == "Bitwise Operations":
        pass


    col1, _, _, _, col2 = st.columns(5)
    col1.page_link("pages/7_Basic_Operations_on_Images.py", label="⬅️**Previous**")
    col2.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label="**Next ➡️**")
    footer()
    