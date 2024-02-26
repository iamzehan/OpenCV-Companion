import streamlit as st

def menu():
    def core_operations_menu():
        st.sidebar.page_link("pages/6_Core_Operations_🔴.py", label="**Core Operations**")
        with st.sidebar.container(border=True):
            st.page_link("pages/7_Basic_Operations_on_Images.py", label="**Basic Operations**")
            st.page_link("pages/8_Arithmetic_Operations_on_Images.py", label ="**Arithmetic Opertations**")
            st.page_link("pages/9_Performance_Measurement_and_Improvement_Techniques.py", label ="**Performance**")
            st.page_link("pages/10_Mathematical_Tools_in_OpenCV.py", label="Math Tools")

    def gui_features_menu():
        st.sidebar.page_link("pages/0_GUI_Features_In_OpenCV.py", label = "**GUI Features in OpenCV**")
        with st.sidebar.container(border=True):    
            st.page_link("pages/1_Getting_Started_with_Images.py", label = "**Getting Started with Images**")
            st.page_link("pages/2_Getting_Started_with_Videos.py", label = "**Getting Started with Videos**")
            st.page_link("pages/3_Drawing_Functions_in_OpenCV.py", label = "**Drawing Functions**")
            st.page_link("pages/4_Mouse_Events.py", label = "**Mouse Events**")
            st.page_link("pages/5_Trackbar.py", label = "**Trackbar**")
    
    st.sidebar.page_link('Getting_Started_with_OpenCV.py', label="Getting Started")
    gui_features_menu()
    core_operations_menu()

