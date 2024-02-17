import streamlit as st 
from utils.gui.footer import footer
if __name__ == '__main__':
    options = st.sidebar.selectbox(label="Navigate: ",options=["Introduction",
                                                               "Image Addition",
                                                               "Image Blending",
                                                               "Bitwise Operations"],
                                   label_visibility="collapsed")
    
    if options == "Introduction":
        st.markdown("""
                    # Arithmetic Operations on Images
                    ## Goal
                    - Learn several arithmetic operations on images like 
                    addition, subtraction, bitwise operations etc.
                    - You will learn these functions : 
                    `cv2.add()`, `cv2.addWeighted()` etc.
                    """)
        
    if options == "Image Addition":
        pass
    if options == "Image Blending":
        pass
    if options == "Bitwise Operations":
        pass
    
    footer()
    