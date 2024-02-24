import streamlit as st
from utils.gui.footer import footer

def main():
    pass

if __name__ == '__main__':
    main()
    col1, _, _, _, col2 = st.columns(5)
    col1.page_link("pages/8_Arithmetic_Operations_on_Images.py", label="⬅️**Previous**")
    col2.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="**Next ➡️**")
    footer()