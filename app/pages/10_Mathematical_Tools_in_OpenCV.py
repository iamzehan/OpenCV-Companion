import streamlit as st
from utils.gui.footer import footer
from  utils.gui.menu import core_operations_menu

def main():
    pass

if __name__ == '__main__':
    core_operations_menu()
    main()
    col1, _, _, _, col2 = st.columns(5)
    col1.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label="⬅️**Previous**")
    footer()