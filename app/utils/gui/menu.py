import streamlit as st

def menu():
    st.sidebar.page_link('Getting_Started_with_OpenCV.py', label="Getting Started")
    st.sidebar.page_link('pages/0_GUI_Features_In_OpenCV_🟢.py', label="GUI Features")
    st.sidebar.page_link('pages/6_Core_Operations_🔴.py', label='Core Operations')

def core_operations_menu():
    st.sidebar.write("Core Operations Menu")
    with st.sidebar.container(border=True):
        st.page_link("pages/6_Core_Operations_🔴.py", label="**Core Operations**")
        st.page_link("pages/7_Basic_Operations_on_Images.py", label="**Basic Operations**")
        st.page_link("pages/8_Arithmetic_Operations_on_Images.py", label ="**Arithmetic Opertations**")
        st.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label ="**Performance**")
        st.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="Math Tools")