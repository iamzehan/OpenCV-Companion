import time
import streamlit as st 
from utils.gui import footer, menu

def main():
    st.title("Contours in OpenCV")
    pages = [{
        "link": "pages/20_1_Contours_GettingStarted.py",
        "label": "Contours: Getting Started",
        "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contour_starting.jpg",
        "description": "Learn to find and draw Contours"
        },
      {
        "link": "pages/20_2_Contour_Features.py",
        "label": "Contour Features",
        "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contour_features.jpg",
        "description": "Learn to find different features of contours like area, perimeter, bounding rectangle etc."
        },
      {
        "link": "pages/20_3_Contour_Properties.py",
        "label": "Contour Properties",
        "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contour_properties.jpg",
        "description": "Learn to find different properties of contours like Solidity, Mean Intensity etc."
        },
      {
        "link": "pages/20_4_Contour_More_Functions.py",
        "label": "Contour: More Functions.py",
        "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contour_defects.jpg",
        "description": "Learn to find convexity defects, pointPolygonTest, match different shapes etc." 
        },
      {
        "link": "pages/20_5_Contours_Hierarchy.py",
        "label": "Contours Hierarchy",
        "image": "https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/contour_hierarchy.jpg",
        "description": "Learn about Contour Hierarchy"
        }]
    
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
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()