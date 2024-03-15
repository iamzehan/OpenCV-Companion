import streamlit as st 
from utils.gui.menu import menu
from utils.gui.footer import footer
from utils.gui.images import ChangingColorSpace

def main():
    ccs = ChangingColorSpace()
    st.title("Changing Colorspaces")
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select: ", options=['Introduction',
                                                'Changing Color-Space',
                                                'Object-Tracking',
                                                ], 
                           horizontal=True,
                           label_visibility="collapsed")
    
    if options == "Introduction":
        st.subheader("Goals")
        st.markdown("""
                    - In this tutorial, you will learn how to convert images from one color-space to another,
                    like BGR $$\leftrightarrow$$ Gray, BGR $$\leftrightarrow$$ HSV etc.
                    - In addition to that, we will create an application which extracts a colored object in a video
                    - You will learn following functions : `cv2.cvtColor()`, `cv2.inRange()` etc.
                    """)
    if options == "Changing Color-Space":
        st.subheader("Changing Color-Space")
        ccs.Changing_Colorspace()
    
    if options == "Object-Tracking":
        st.subheader("Object-Tracking")
        ccs.uploader()
        ccs.Object_Tracking()

if __name__ == '__main__':
    st.set_page_config("Changing Colorspaces", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu()
    main()
    footer()