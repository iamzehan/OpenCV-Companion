import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Contours: Getting Started")
    getting_started= images.Contours.GettingStarted()
    
    with st.container(border=True):
        
        st.subheader("Topic")
        options = st.radio("Options: ",
                           options= ["Introduction",
                                     "What are contours?",
                                     "How to draw the contours?",
                                     "Contour Approximation Method"],
                           horizontal=True,
                           label_visibility="collapsed")
        
    with st.container(border=True):
        if options == "Introduction":
            getting_started.Introduction()
        
        elif options == "What are contours?":
            getting_started.What_are_Contours()
        
        elif options == "How to draw the contours?":
            getting_started.How_to_Draw()
            
        elif options == "Contour Approximation Method":
            getting_started.Contour_Approx_Method()

if __name__ == "__main__":
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
