import streamlit as st
from PIL import ImageColor
from utils.gui.drawing import (
    Draw_Line,
    Draw_Rectangle,
    Draw_Circle,
    Draw_Ellipse,
    Draw_Polygon,
    Draw_Text
)

if __name__ == "__main__":
    st.set_page_config(page_icon="app\\assets\\OpenCV_Logo_with_text.png",
                       page_title="Drawing Geometric Shapes with OpenCV")
    st.markdown("""
                ## Drawing Geometric Shapes with OpenCV 📐📏 🟥🔴🔺
                
                ## Goal
                Learn to draw different geometric shapes with OpenCV. The functions covered include 
                `cv.line()`, `cv.circle()`, `cv.rectangle()`, `cv.ellipse()`, `cv.putText()`, and more.

                ## Code
                In all the above functions, you will find common arguments as given below:

                - `img`: The image where you want to draw the shapes.
                - `color`: Color of the shape. For BGR, pass it as a tuple, e.g., (255, 0, 0) for blue. 
                For grayscale, just pass the scalar value.
                - `thickness`: Thickness of the line or circle, etc. 
                If -1 is passed for closed figures like circles, it will fill the shape. The default thickness is 1.
                - `lineType`: Type of line, whether 8-connected, anti-aliased line, etc. By default, it is 8-connected. 
                `cv.LINE_AA` gives an anti-aliased line, which looks great for curves.
    """)

    st.sidebar.subheader("Drawing Options")
    options = st.sidebar.selectbox(label="Select:",
                                   options=["Drawing Line",
                                            "Drawing Rectangle",
                                            "Drawing Circle",
                                            "Draw Ellipse",
                                            "Drawing Polygon",
                                            "Adding Texts"],
                                   label_visibility="collapsed")
    
    if options == "Drawing Line":
        Draw_Line()
        
    if options == "Drawing Rectangle":
        Draw_Rectangle()
        
    if options == "Drawing Circle":
        Draw_Circle()
        
    if options == "Draw Ellipse":
        Draw_Ellipse()
    
    if options == "Drawing Polygon":
        Draw_Polygon()
        
    if options == "Adding Texts":
        Draw_Text()