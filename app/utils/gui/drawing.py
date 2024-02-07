import streamlit as st
from PIL import ImageColor
from utils.opencv.drawing import draw_line, draw_rectangle, draw_circle

# Draw Line Parameters
def Draw_Line():
    
    with st.container(border=True):
        
        st.subheader("Drawing Line")
        st.markdown("""To draw a line, you need to pass starting and ending coordinates of line. 
                    We will create a black image and draw a blue line on it from top-left to bottom-right corners.""")
        st.info("Feel free to fiddle around with the parameters")
        
        with st.container(border=True):
            st.markdown("<center style='color:red'><h3>Parameters</h3></center>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            col1.markdown("<center>Start</center>", unsafe_allow_html=True)
            start_x, start_y =  col1.slider("`x - coordinate`", max_value=512),\
                                col1.slider("` y - coordinate`", max_value= 512)
            start = (start_x, start_y)
            
            col1.markdown("<center>End</center>", unsafe_allow_html=True)
            end_x, end_y =  col1.slider("`x - coordinate`", value=511),\
                            col1.slider("` y - coordinate`", value=511)
            end = (end_x, end_y)
            
            col1.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = col1.color_picker("Pick a color",value="#ff0000", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            col1.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness = col1.slider("Thickness",value=5, min_value=1, max_value=10, label_visibility="hidden")
            
            col2.markdown("<center>Output</center>", unsafe_allow_html=True)
            col2.image(draw_line(start, end, color, thickness),'Draw Line')
                
        with st.container(border=True):
            st.markdown("### Code")
            st.code(f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal red line with thickness of 5 px
                    cv.line(img,{start},{end},{color},{thickness})
        """)


def Draw_Rectangle():
    
    with st.container(border=True):
        
        st.subheader("Drawing Rectangle")
        st.markdown("""
                    To draw a rectangle, you need the top-left corner and bottom-right corner of the rectangle. 
                    This time, we will draw a green rectangle at the top-right corner of the image.
                    """)
        st.info("Feel free to fiddle around with the parameters")
        
        with st.container(border=True):
            st.markdown("<center style='color:red'><h3>Parameters</h3></center>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            col1.markdown("<center>Top-left</center>", unsafe_allow_html=True)
            top_left_x, top_left_y = col1.number_input("`x - coordinate`",value=384,max_value=512),\
                                     col1.number_input("` y - coordinate`", value=0, max_value= 512)
            top_left = (top_left_x, top_left_y)
            
            col1.markdown("<center>Bottom-Right</center>", unsafe_allow_html=True)
            bottom_right_x, bottom_right_y = col1.number_input("`x - coordinate`", value=510, max_value=512),\
                                             col1.number_input("` y - coordinate`", value=128, max_value= 512)
            bottom_right = (bottom_right_x, bottom_right_y)
            
            col1.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = col1.color_picker("Pick a color",value="#00ff00", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            col1.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness = col1.number_input("Thickness",value=5, min_value=1,
                                        max_value=10, label_visibility="hidden")
            
            if col1.checkbox("Fill rectangle"): thickness=-1
            
            col2.markdown("<center>Output</center>", unsafe_allow_html=True)
            col2.image(draw_rectangle(top_left, 
                                      bottom_right, 
                                      color, thickness),
                       'Draw Rectangle')
            
                
        with st.container(border=True):
            st.markdown("### Code")
            st.code(f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal blue line with thickness of 5 px
                    cv.rectangle(img,{top_left},{bottom_right},{color},{thickness})
        """)
            
def Draw_Circle():
    with st.container(border=True):
        st.subheader("Drawing Circle")
        st.markdown("""
                    To draw a circle, you need its center coordinates and radius.
                    """)
        st.info("Feel free to fiddle around with the parameters")
        
        with st.container(border=True):
            st.markdown("<center style='color:red'><h3>Parameters</h3></center>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            col1.markdown("<center>Center</center>", unsafe_allow_html=True)
            center_x, center_y = col1.slider("`x - coordinate`",value=447,max_value=512),\
                                 col1.slider("` y - coordinate`", value=63, max_value= 512)
            center = (center_x, center_y)
            
            col1.markdown("<center>Radius</center>", unsafe_allow_html=True)
            radius= col1.number_input("`r`", value=63, max_value=512)
            
            col1.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = col1.color_picker("Pick a color",value="#00ff00", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            col1.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness=col1.number_input("Thickness",value=-1,
                                        max_value=10, label_visibility="hidden")
            
            if col1.checkbox("Fill Circle", value=True): thickness=-1
            
            col2.markdown("<center>Output</center>", unsafe_allow_html=True)
            col2.image(draw_circle( center, 
                                    radius, 
                                    color,
                                    thickness ),
                       'Draw Circle')
            
                
        with st.container(border=True):
            st.markdown("### Code")
            st.code(f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal blue line with thickness of 5 px
                    cv.circle(img,{center},{radius},{color},{thickness})
        """)