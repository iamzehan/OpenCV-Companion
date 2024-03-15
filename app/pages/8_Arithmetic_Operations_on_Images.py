import streamlit as st 
from utils.gui import footer, menu, images

def main():
    st.markdown("# Arithmetic Operations on Images")
    arithmeticOps = images.ArithmeticOperations() 
    
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio(label="Navigate: ",
                           options=["Introduction",
                                    "Image Addition",
                                    "Image Blending",
                                    "Bitwise Operations"],
                            horizontal=True,
                            label_visibility="collapsed")
    
    if options == "Introduction":
        st.markdown("""
                    ## Goal
                    - Learn several arithmetic operations on images like 
                    addition, subtraction, bitwise operations etc.
                    - You will learn these functions : 
                    `cv2.add()`, `cv2.addWeighted()` etc.
                    """)
        
    if options == "Image Addition":
        arithmeticOps.uploader(multiple=True)
        arithmeticOps.Image_Addition()
    if options == "Image Blending":
        arithmeticOps.uploader(multiple=True)
        arithmeticOps.Image_Blending()
    if options == "Bitwise Operations":
        arithmeticOps.uploader(multiple=True)
        arithmeticOps.Bitwise_Operations()

if __name__ == "__main__":
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Arithmetic Operations on Images")
    menu.menu()
    main()
    footer.footer()
    