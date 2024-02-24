import streamlit as st
from utils.gui.footer import footer

def main():
    pass

if __name__ == '__main__':
    main()
    col1, _, _, _, col2 = st.columns(5)
    col1.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label="⬅️**Previous**")
    footer()