import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histograms - 2: Histogram Equalization")
    hist2 = images.Histograms.Histograms2()
    
    functions = {
        "Introduction" : hist2.Introduction,
        "Theory" : hist2.Theory,
        "Histograms Equalization" : hist2.HistogramsEqualization,
        "CLAHE": hist2.CLAHE
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
