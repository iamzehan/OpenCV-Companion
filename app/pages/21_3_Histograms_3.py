import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histograms - 3 : 2D Histograms")

    st.markdown("Learn to find and plot 2D Histograms")

if __name__ == "__main__":
    st.set_page_config("Histograms in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
