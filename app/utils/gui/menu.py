import streamlit as st

def menu():
    st.sidebar.page_link('Getting_Started_with_OpenCV.py', label="Getting Started")
    st.sidebar.page_link('pages/0_GUI_Features_In_OpenCV_🟢.py', label="GUI Features")
    st.sidebar.page_link('pages/6_Core_Operations_🔴.py', label='Core Operations')

if __name__ == '__main__':
    menu()
    
    