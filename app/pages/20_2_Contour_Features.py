import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Contour Features")
    features = images.Contours.Features()
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio(label="Options: ",
                        options=["Introduction",
                                 "Moments"],
                        horizontal = True,
                        label_visibility="collapsed")
    
    if options == "Introduction":
        features.Introduction()
    elif options == "Moments":
        features.Moments()

if __name__ == "__main__":
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
