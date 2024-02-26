import streamlit as st
from utils.gui.footer import footer
from  utils.gui.menu import menu

def main():
    pass

if __name__ == '__main__':
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Mathematical Tools")
    menu()
    main()
    footer()