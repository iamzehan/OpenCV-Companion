import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histograms - 2: Histogram Equalization")

    st.markdown("Learn to Equalize Histograms to get better contrast for images")

if __name__ == "__main__":
    st.set_page_config("Histograms in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
