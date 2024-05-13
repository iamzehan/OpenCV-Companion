import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histogram - 4 : Histogram Backprojection")
    hist4 = images.Histograms.Histograms4()
    functions = {
        "Introduction" : hist4.Introduction,
        "Algorithm in Numpy" : hist4.Algorithm_in_Numpy,
        "Backprojection in OpenCV": hist4.Backprojection_in_OpenCV
     }
    
    with st.container(border=True):
                st.subheader("Topics")
                options = st.radio(label="Options: ",
                                options=list(functions.keys()),
                                horizontal = True,
                                label_visibility="collapsed")
            
    if options:
        functions[options]()

if __name__ == "__main__":
    st.set_page_config("Histograms in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
