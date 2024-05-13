import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histograms - 3 : 2D Histograms")

    st.markdown("Learn to find and plot 2D Histograms")
    hist3 = images.Histograms.Histograms3()
    functions = {
        "Introduction" : hist3.Introduction,
        "2D Histograms in OpenCV" : hist3.Histograms_2D_OpenCV,
        "2D Histograms in Numpy" : hist3.Histograms_2D_Numpy,
        "Plotting 2D Histograms" : hist3.Plotting_2D_Histograms
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
