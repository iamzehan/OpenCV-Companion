import time
import streamlit as st 
from utils.gui import footer, menu

def main():
    st.title("Histograms in OpenCV")
    pages= [
        {
            "link": "pages/21_1_Histograms_1.py",
            "label": "Histograms - 1 : Find, Plot, Analyze !!!",
            "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/histograms_1d.jpg",
            "description": "Learn to find and draw Contours"
            },
        {
            "link": "pages/21_2_Histograms_2.py",
            "label": "Histograms - 2: Histogram Equalization",
            "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/histograms_equ.jpg",
            "description": "Learn to Equalize Histograms to get better contrast for images"
            },
        {
            "link": "pages/21_3_Histograms_3.py",
            "label": "Histograms - 3 : 2D Histograms",
            "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/histograms_2d.jpg",
            "description": "Learn to find and plot 2D Histograms"
            },
        {
            "link": "pages/21_4_Histograms_4.py",
            "label": "Histogram - 4 : Histogram Backprojection",
            "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/histograms_bp.jpg",
            "description": "Learn histogram backprojection to segment colored objects"
            }
        ]
    
    for i in range(len(pages)):
        with st.container(border=True):
            with st.spinner("Please wait.."):
                time.sleep(0.2)
                link, label, image, description= pages[i]['link'], pages[i]['label'], pages[i]['image'], pages[i]['description']
                st.page_link(f"{link}", label=f"{i+1} . **{label}**", use_container_width=True)
                col1, col2, col3 = st.columns([2,8,2])
                col1.image(f"{image}", use_column_width=True)
                col2.markdown(f"{description}")
                col3.page_link(f"{link}", label="**Learn**➡️", use_container_width=True)

if __name__ == '__main__':
    st.set_page_config("Histograms in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()