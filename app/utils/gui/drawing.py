import streamlit as st
from PIL import ImageColor
from utils.opencv.drawing import (draw_line,
                                  draw_rectangle,
                                  draw_circle,
                                  draw_ellipse,
                                  draw_polygon,
                                  draw_text)

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
        st.markdown("""
                    #### Note
                    > If third argument is False, you will get a polylines joining all the points, not a closed shape.
                    `cv.polylines()` can be used to draw multiple lines. Just create a list of all the lines you want to draw and pass it to the function.
                    All lines will be drawn individually. It is a much better and faster way to draw a group of lines than calling cv.line() for each line.""")
    
    live = False
    points = st.sidebar.number_input("How many points for your Polygon?", value=0)
    st.sidebar.info("Feel free to fiddle around with the parameters")
    
    st.sidebar.subheader("Parameters")
    with st.sidebar.container(border=True):
        
        pts = []
        
        if points:
            live=True
            for i in range(points):
                st.markdown(f"Coordinates for point : `{i+1}`")
                x = st.slider(f"$x_{i+1}$:", key=f"x{i}")
                y = st.slider(f"$y_{i+1}$:", key=f"y{i}")
                pts.append([x, y])
                
        join = st.checkbox("Join?", value=True)
        color = st.color_picker("Color:", value="#00ffff")
        color = ImageColor.getcolor(f'{color}','RGB')
        
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
    
    def write_code(text='OpenCV',
              position=(10, 500),
              font='cv.FONT_HERSHEY_SIMPLEX',
              font_scale=4,
              color = (255,255,255),
              thickness=2,
              lineType='cv.LINE_AA'):
        font_dict = {
    'HERSHEY_SIMPLEX': 'cv.FONT_HERSHEY_SIMPLEX',
    'HERSHEY_PLAIN': 'cv.FONT_HERSHEY_PLAIN',
    'HERSHEY_DUPLEX': 'cv.FONT_HERSHEY_DUPLEX',
    'HERSHEY_COMPLEX': 'cv.FONT_HERSHEY_COMPLEX',
    'HERSHEY_TRIPLEX': 'cv.FONT_HERSHEY_TRIPLEX',
    'HERSHEY_COMPLEX_SMALL': 'cv.FONT_HERSHEY_COMPLEX_SMALL',
    'HERSHEY_SCRIPT_SIMPLEX': 'cv.FONT_HERSHEY_SCRIPT_SIMPLEX',
    'HERSHEY_SCRIPT_COMPLEX': 'cv.FONT_HERSHEY_SCRIPT_COMPLEX'
}
        line_type_dict = {
    'LINE_AA': 'cv.LINE_AA',
    'LINE_4': 'cv.LINE_4',
    'LINE_8': 'cv.LINE_8'
}
        return f"""
            import numpy as np
            import cv2 as cv
            # Create a black image
            img = np.zeros((512,512,3), np.uint8)
            cv.putText(img, '{text}', {position}, {font_dict[font]},
                        {font_scale}, {color}, {thickness}, {line_type_dict[lineType]})
            """
                
    with st.container(border=True):
        st.subheader("Adding Text to Images")
        st.markdown("""
                    To put texts in images, you need specify following things.

                    - Text data that you want to write
                    - Position coordinates of where you want put it (i.e. bottom-left corner where data starts).
                    - Font type (Check cv.putText() docs for supported fonts)
                    - Font Scale (specifies the size of font)
                    - regular things like color, thickness, lineType etc. For better look, lineType = cv.LINE_AA is recommended.
                    
                    We will write OpenCV on our image in white color.""")
                
        image_container=st.empty()
        code_container = st.empty().container(border=True)
        
    st.sidebar.info("Feel free to fiddle around with the parameters")
    
    st.sidebar.subheader("Parameters")
    
    with st.sidebar.container(border=True):
        text=st.text_input("Add text", placeholder="Add some text...")
        position = st.slider("Position - `x`", value=10, max_value=500),\
                    st.slider("Position - `y`", value=500)
        
        font= st.selectbox(label="Select font: ", options=['HERSHEY_SIMPLEX', 
                                                   'HERSHEY_PLAIN', 
                                                   'HERSHEY_DUPLEX', 
                                                   'HERSHEY_COMPLEX', 
                                                   'HERSHEY_TRIPLEX', 
                                                   'HERSHEY_COMPLEX_SMALL', 
                                                   'HERSHEY_SCRIPT_SIMPLEX', 
                                                   'HERSHEY_SCRIPT_COMPLEX'])
        
        font_scale=st.number_input(label="Font Scale", value=4, min_value=1, max_value=10)
        
        color = st.color_picker("Color:", value="#fff")
        color = ImageColor.getcolor(f'{color}','RGB')
        
        thickness=st.number_input(label="Thickness", value=2, min_value=1, max_value=10)
        lineType= st.selectbox(label="Select Line Type: ",
                               options=['LINE_AA', 
                                        'LINE_4', 
                                        'LINE_8'])
        
        live = st.checkbox("Edit Live?", value=False)
        add_text=st.button("Add Text", type="primary", use_container_width=True)
        
    with image_container.container(border=True):
        st.markdown("<center> Output </center>", unsafe_allow_html=True)
        if add_text or live: 
            st.image(draw_text(text, position, font, font_scale, color, thickness, lineType), caption="Drawing Polygon", use_column_width=True)
            code_container.markdown("### Code")
            code_container.code(write_code(text, position, font, font_scale, color, thickness, lineType))
        else:
            st.image(draw_text(), caption="Drawing Polygon", use_column_width=True)
            code_container.markdown("### Code")
            code_container.code(write_code())