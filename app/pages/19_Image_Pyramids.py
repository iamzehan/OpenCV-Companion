import streamlit as st 
from utils.gui import footer, menu, images

def main():
    img_pyr = images.ImagePyramids()
    st.title("Image Pyramids")
    
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Options: ", 
                        options=["Introduction", "Theory", "Image Blending using Pyramids"],
                        horizontal=True,
                        label_visibility="collapsed")
    
    with st.container(border=True):
        
        if options == "Introduction":
            img_pyr.Introduction()
            
        elif options=="Theory":
            img_pyr.Theory()
        
        elif options == "Image Blending using Pyramids":
            img_pyr.ImageBlending()

if __name__ == '__main__':
    st.set_page_config("Image Pyramids", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()