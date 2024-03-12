import streamlit as st 
from utils.gui import footer, menu, images

def main():
    morph_trans = images.MorphologicalTransformation()
    st.markdown("""
                # Morphological Transformations

                ## Goal

                In this chapter,
                We will learn different morphological operations like Erosion, Dilation, Opening, Closing, etc.
                We will see different functions like:
                - `cv2.erode()`
                - `cv2.dilate()`
                - `cv2.morphologyEx()`

                """)
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Select Function:", ["Theory", "Erosion", "Dilation", "Opening", "Closing", 
                                            "MorphGradient", "TopHat", "BlackHat", "StructuringElement"], 
                        horizontal=True, label_visibility="collapsed")

    if options == "Theory":
        morph_trans.Theory()
    elif options == "Erosion":
        morph_trans.Erosion()
    elif options == "Dilation":
        morph_trans.Dilation()
    elif options == "Opening":
        morph_trans.Opening()
    elif options == "Closing":
        morph_trans.Closing()
    elif options == "MorphGradient":
        morph_trans.MorphGradient()
    elif options == "TopHat":
        morph_trans.TopHat()
    elif options == "BlackHat":
        morph_trans.BlackHat()
    elif options == "StructuringElement":
        morph_trans.StructuringElement()

if __name__ == '__main__':
    st.set_page_config("Morphological Transformations", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()