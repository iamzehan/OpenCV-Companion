import streamlit as st 
from utils.gui.footer import footer
from  utils.gui.menu import menu
from utils.gui.images import ArithmeticOperations

if __name__ == '__main__':
    menu()
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Arithmetic Operations on Images")
    with st.sidebar.container(border=True):
        st.subheader("Topics")
        options = st.radio(label="Navigate: ",options=["Introduction",
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
        arithmeticOps.side_bar(multiple=True)
        arithmeticOps.Image_Blending()
    if options == "Bitwise Operations":
        arithmeticOps.side_bar(multiple=True)
        arithmeticOps.Bitwise_Operations()

    footer()
    