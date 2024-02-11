import streamlit as st
from PIL import ImageColor
from utils.opencv.drawing import (draw_line,
                                  draw_rectangle,
                                  draw_circle,
                                  draw_ellipse,
                                  draw_polygon)

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
            st.markdown("<center><h3>Parameters</h3></center>", unsafe_allow_html=True)
            
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
        
        
        with st.sidebar.container(border=True):
            st.markdown("<center>Center</center>", unsafe_allow_html=True)
            center_x, center_y = st.slider("`x - coordinate`",value=447,max_value=512),\
                                st.slider("` y - coordinate`", value=63, max_value= 512)
            center = (center_x, center_y)
            
            st.markdown("<center>Radius</center>", unsafe_allow_html=True)
            radius= st.number_input("`r`", value=63, max_value=512)
            
            st.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = st.color_picker("Pick a color",value="#00ff00", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            st.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness=st.number_input("Thickness",value=2,
                                        max_value=10, label_visibility="hidden")
            
            if st.checkbox("Fill Circle", value=True): thickness=-1
        
        with st.container(border=True): 
            st.markdown("<center>Output</center>", unsafe_allow_html=True)
            st.image(draw_circle( center, 
                                    radius, 
                                    color,
                                    thickness ),
                       'Draw Circle',
                       width=200, use_column_width=True)
            
                
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

def Draw_Ellipse():
    
    with st.container(border=True):
        st.subheader("Drawing Ellipse")
        st.markdown("""
                    To draw the ellipse, we need to pass several arguments.
                    One argument is the center location (x,y).
                    Next argument is axes lengths (major axis length, minor axis length).
                    `angle` is the angle of rotation of ellipse in anti-clockwise direction. 
                    `startAngle` and `endAngle` denotes the starting and ending of ellipse 
                    arc measured in clockwise direction from major axis. 
                    i.e. giving values 0 and 360 gives the full ellipse.
                    For more details, check the documentation of cv.ellipse().
                    Below example draws a half ellipse at the center of the image.
                    """)
        st.sidebar.info("Feel free to fiddle around with the parameters")
        
        with st.container(border=True):
            
            with st.sidebar.container(border=True):
            
                st.markdown("<center>Center</center>",
                            unsafe_allow_html=True)
                center_x, center_y = st.slider("`x - coordinate`",value=447,max_value=512),\
                                    st.slider("` y - coordinate`", value=63, max_value= 512)
                center = (center_x, center_y)
                
                st.markdown("<center>Axes Length</center>", unsafe_allow_html=True)
                axes_length= st.slider("`axes_length - major`", value=100, max_value=512),\
                            st.slider("`axes_length - minor`", value=50, max_value=512)
                
                st.markdown("<center>Angle</center>", unsafe_allow_html=True)
                angle = st.slider("`angle`", value = 0, max_value=360)
                
                st.markdown("<center>Start Angle</center>", unsafe_allow_html=True)
                start_angle = st.slider("`start_angle`", value = 0, max_value=360)
                
                st.markdown("<center>End Angle</center>", unsafe_allow_html=True)
                end_angle = st.slider("`end_angle`", value = 180, max_value=360)
                
                st.markdown("<center>Color</center>", unsafe_allow_html=True)
                color = st.color_picker("Pick a color",value="#ff0000", label_visibility="hidden")
                color = ImageColor.getcolor(f'{color}','RGB')
                
                st.markdown("<center>Thickness</center>", unsafe_allow_html=True)
                thickness=st.slider("Thickness",value=2, min_value=-1,
                                            max_value=10, label_visibility="hidden")
                
                if st.checkbox("Fill Circle", value=True): thickness=-1
            
            st.markdown("<center>Output</center>", unsafe_allow_html=True)
            st.image(draw_ellipse( center, 
                                    axes_length,
                                    angle,
                                    start_angle,
                                    end_angle,
                                    color,
                                    thickness ),
                       'Draw Ellipse',
                       use_column_width=True)
            
                
        with st.container(border=True):
            st.markdown("### Code")
            st.code(f"""
                import numpy as np
                import cv2 as cv
                # Create a black image
                img = np.zeros((512,512,3), np.uint8)
                # Draw a diagonal blue line with thickness of 5 px
                cv.ellipse(img,{center}, {axes_length}, {angle}, {start_angle}, {end_angle}, {color}, {thickness})
        """)

def Draw_Polygon():
    
    def write_code(pts=[[10,5],[20,30],[70,20],[50,10]], join=True, color=(255,255,255)):
        return f"""
                import numpy as np
                import cv2 as cv
                # Create a black image
                img = np.zeros((512,512,3), np.uint8)
                pts = np.array({pts}, np.int32)
                pts = pts.reshape((-1,1,2))
                cv.polylines(img,[pts], {join}, {color})
                """
    with st.container(border=True):
        st.subheader("Drawing Polygons")
        st.markdown("""
                    To draw a polygon, first you need coordinates of vertices.
                    Make those points into an array of shape ROWSx1x2 where ROWS
                    are number of vertices and it should be of type int32. 
                    Here we draw a small polygon of with four vertices in yellow color.
                    """)
                
        image_container=st.empty()
        code_container = st.empty().container(border=True)
    
    st.sidebar.info("Feel free to fiddle around with the parameters")
    with st.sidebar.container(border=True):
        
        points = st.number_input("How many points for your Polygon?", value=0)
        pts = []
        
        if points:
            for i in range(points):
                st.markdown(f"Coordinates for point : `{i+1}`")
                x = st.slider(f"$x_{i+1}$:", key=f"x{i}")
                y = st.slider(f"$y_{i+1}$:", key=f"y{i}")
                pts.append([x, y])
                
        join = st.checkbox("Join?", value=True)
        color = st.color_picker("Color:", value="#00ffff")
        color = ImageColor.getcolor(f'{color}','RGB')
        live = st.checkbox("Generate Polygon live?", value = False)
    
    with image_container.container(border=True):
        st.markdown("<center> Output </center>", unsafe_allow_html=True)
        if live: 
            st.image(draw_polygon(pts, join, color), caption="Drawing Polygon", use_column_width=True)
            code_container.markdown("### Code")
            code_container.code(write_code(pts, join, color))
        else:
            st.image(draw_polygon(), caption="Drawing Polygon", use_column_width=True)
            code_container.markdown("### Code")
            code_container.code(write_code())
        

def Draw_Text():
    pass