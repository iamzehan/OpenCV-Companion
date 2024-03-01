import streamlit as st 
from utils.gui import footer, menu

def main():
    pass

if __name__ == '__main__':
    st.set_page_config("Hough Circle Transform", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()