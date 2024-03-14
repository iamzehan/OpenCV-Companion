import streamlit as st
from utils.gui import footer, menu, images

def main():
    st.title("Contours Hierarchy")

    st.markdown("## contour_5: Contours Hierarchy")
    st.write("Content for contour_5 goes here.")

if __name__ == "__main__":
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()