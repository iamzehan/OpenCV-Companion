import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Contour Features")
    features = images.Contours.Features()
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio(label="Options: ",
                        options=["Introduction",
                                 "Moments",
                                 "Contour Area",
                                 "Contour Perimeter",
                                 "Contour Approximation",
                                 "Convex Hul",
                                 "Checking Convexity",
                                 "Bounding Rectangle",
                                 "Minimum enclosing Circle",
                                 "Fitting an Ellipse",
                                 "Fitting a Line"],
                        horizontal = True,
                        label_visibility="collapsed")

    if options == "Introduction":
        features.Introduction()
    elif options == "Moments":
        features.Moments()
    elif options == "Contour Area":
        features.Contour_Area()
    elif options == "Contour Perimeter":
        features.Contour_Perimeter()
    elif options == "Contour Approximation":
        features.Contour_Approximation()
    elif options == "Convex Hul":
        features.Convex_Hull()
    elif options == "Checking Convexity":
        features.Checking_Convexity()
    elif options == "Bounding Rectangle":
        features.Bounding_Rectangle()
    elif options == "Minimum enclosing Circle":
        features.Minimum_Enclosing_Circle()
    elif options == "Fitting an Ellipse":
        features.Fitting_an_Ellipse()
    elif options == "Fitting a Line":
        features.Fitting_a_Line()

if __name__ == "__main__":
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
