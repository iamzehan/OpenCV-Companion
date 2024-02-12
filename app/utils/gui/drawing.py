import streamlit as st
from PIL import ImageColor
from utils.opencv.drawing import \
    (draw_line,
    draw_rectangle,
    draw_circle,
    draw_ellipse,
    draw_polygon,
    draw_text)

def Draw_Line():
    """
    The `Draw_Line` function allows users to draw a line on an image by specifying the starting and
    ending coordinates, color, and thickness of the line.
    """
    
    def write_code(start:tuple, end:tuple, color:tuple, thickness:int) -> str:
        return f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal red line with thickness of 5 px
                    cv.line(img, {start}, {end}, {color[::-1]}, {thickness})
                    cv.imshow('Output', img)
        """
    
    defaults=[(0, 0), (511, 511), (255, 0, 0), 5]
    
    main_container = st.empty().container(border=True)
    main_container.subheader("Drawing Line")
    main_container.markdown("""
                To draw a line, you need to pass starting and ending coordinates of line. 
                We will create a black image and draw a blue line on it from top-left to 
                bottom-right corners.
                """)
    
    image_container = main_container.container(border=True)
    code_container = main_container.container(border=True)
    
    image_container.subheader("Output")
    code_container.subheader("Code")
    
    with st.sidebar:
        
        st.markdown("<center style='color:red'><h5>Parameters</h5></center>", unsafe_allow_html=True)
        st.info("Change Parameters to see differences")
            
        with st.container(border=True):
            st.markdown("<center>Start</center>", unsafe_allow_html=True)
            start = st.slider("`x - coordinate`", max_value=512),\
                    st.slider("` y - coordinate`", max_value= 512)
            
            st.markdown("<center>End</center>", unsafe_allow_html=True)
            end = st.slider("`x - coordinate`", value=511),\
                  st.slider("` y - coordinate`", value=511)
            
            st.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = st.color_picker("Pick a color",value="#ff0000", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            st.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness = st.slider("Thickness",value=5, min_value=1, max_value=10, label_visibility="hidden")

    if [start, end, color, thickness] != defaults: 
        image_container.image(draw_line(start, end, color, thickness),'Draw Line', use_column_width=True)
        code_container.success("Your code")
        code_container.code(write_code(start, end, color, thickness))
        image_container.success("Your Output")
    else:
        image_container.image(draw_line(),'Draw Line', use_column_width=True)
        code_container.info("Example Code")
        code_container.code(write_code(*defaults))
        image_container.info("Example Output")


def Draw_Rectangle():
    """
    The `Draw_Rectangle` function allows the user to draw a rectangle on an image by specifying the
    top-left and bottom-right coordinates, color, and thickness.
    """
    
    def write_code(top_left, bottom_right, color, thickness):
        return f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal blue line with thickness of 5 px
                    cv.rectangle(img,{top_left},{bottom_right},{color[::-1]},{thickness})
                    cv.imshow('Output', img)
        """
        
    defaults = [(384, 0), (510, 128), (0, 255, 0), 5]
    
    main_container = st.empty().container(border=True)
    main_container.subheader("Drawing Rectangle")
    main_container.markdown("""
                To draw a rectangle, you need the top-left corner and bottom-right corner of the rectangle. 
                This time, we will draw a green rectangle at the top-right corner of the image.
                """)
    
    image_container = main_container.container(border=True)
    code_container = main_container.container(border=True)
    
    with st.sidebar:
        
        st.markdown("<center><h5>Parameters</h5></center>", unsafe_allow_html=True)
        st.info("Change Parameters to see differences")
        
        with st.container(border=True):
            st.markdown("<center>Top-left</center>", unsafe_allow_html=True)
            
            top_left = st.number_input("`x - coordinate`",value=384,max_value=512),\
                                        st.number_input("` y - coordinate`", value=0, max_value= 512)
            
            st.markdown("<center>Bottom-Right</center>", unsafe_allow_html=True)
            bottom_right = st.number_input("`x - coordinate`", value=510, max_value=512),\
                                                st.number_input("` y - coordinate`", value=128, max_value= 512)
            
            st.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = st.color_picker("Pick a color",value="#00ff00", label_visibility="hidden")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            st.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness = st.number_input("Thickness",value=5, min_value=1,
                                        max_value=10, label_visibility="hidden")
            
            if st.checkbox("Fill rectangle"): thickness=-1
        
    image_container.subheader("Output")
    code_container.subheader("Code")
            
    if [top_left, bottom_right, color, thickness] != defaults:
        image_container.image(draw_rectangle(top_left, 
                                bottom_right, 
                                color, thickness),
                'Draw Rectangle',
                use_column_width=True)
        image_container.success("Your Output")
        code_container.code(write_code(top_left, bottom_right, color, thickness))
        code_container.success("Your code")
    
    else:
        image_container.image(draw_rectangle(),
                'Draw Rectangle',
                use_column_width=True)
        image_container.info("Example Output")
        
        code_container.code(write_code(*defaults))
        code_container.info("Example code")
            
def Draw_Circle():
    """
    The `Draw_Circle` function allows users to draw a circle by specifying the center coordinates,
    radius, color, and thickness.
    """
    
    def write_code(center, radius, color, thickness):
        return f"""
                    import numpy as np
                    import cv2 as cv
                    # Create a black image
                    img = np.zeros((512,512,3), np.uint8)
                    # Draw a diagonal blue line with thickness of 5 px
                    cv.circle(img,{center},{radius},{color[::-1]},{thickness})
                    cv.imshow('Output', img)
        """
        
    defaults = [(447, 63),63,(0, 255, 0),-1]
     
    main_container = st.container(border=True)
    main_container.subheader("Drawing Circle")
    main_container.markdown("""
                To draw a circle, you need its center coordinates and radius.
                """)
    main_container.info("Change Parameters to see differences")
    
    image_container = main_container.container(border=True)
    code_container = main_container.container(border=True)
    
    with st.sidebar:
        
        st.markdown("<center><h5>Parameters</h5></center>", unsafe_allow_html=True)
        st.info("Change Parameters to see differences")
        
        with st.container(border=True):
            st.markdown("<center>Center</center>", unsafe_allow_html=True)
            center_x, center_y = st.slider("`x - coordinate`",value=447,max_value=512),\
                                st.slider("` y - coordinate`", value=63, max_value= 512)
            center = (center_x, center_y)
            
            st.markdown("<center>Radius</center>", unsafe_allow_html=True)
            radius= st.number_input("`r`", value=63, max_value=512, label_visibility="collapsed")
            
            st.markdown("<center>Color</center>", unsafe_allow_html=True)
            color = st.color_picker("Pick a color",value="#00ff00", label_visibility="collapsed")
            color = ImageColor.getcolor(f'{color}','RGB')
            
            st.markdown("<center>Thickness</center>", unsafe_allow_html=True)
            thickness = st.number_input("Thickness",
                                        value=2, 
                                        min_value=-1,
                                        max_value=10, 
                                        label_visibility="collapsed")
               
            fill = st.checkbox("Fill Circle?")
            
            if fill: thickness = -1
            
    
    image_container.markdown("<center>Output</center>", unsafe_allow_html=True)
    code_container.markdown("### Code")
    
    if [center, radius, color, thickness] != defaults:
    
        image_container.image(draw_circle( center, 
                                radius, 
                                color,
                                thickness ),
                    'Draw Circle',
                    width=200, use_column_width=True)
        image_container.success("Your Output")
        
        code_container.code(write_code(center, radius, color, thickness))
        code_container.success("Your Code")
        
    else:
        
        image_container.image(draw_circle(),
                    'Draw Circle',
                    width=200, use_column_width=True)
        image_container.info("Example Output")
        
        code_container.code(write_code(*defaults))
        code_container.info("Example Code")

def Draw_Ellipse():
    """
    The function `Draw_Ellipse()` allows users to draw an ellipse on an image using OpenCV in Python,
    with customizable parameters such as center location, axes lengths, angle, start angle, end angle,
    color, and thickness.
    """
    
    def write_code( center:tuple, 
                    axes_length:tuple,
                    angle:int,
                    start_angle:int,
                    end_angle:int,
                    color:tuple,
                    thickness:int) -> str:
        
        return f"""
            import numpy as np
            import cv2 as cv
            # Create a black image
            img = np.zeros((512,512,3), np.uint8)
            # Draw a diagonal blue line with thickness of 5 px
            cv.ellipse(img,{center}, {axes_length}, {angle}, {start_angle}, {end_angle}, {color[::-1]}, {thickness})
            cv.imshow('Output', img)
        """
        
    
    defaults=[(447, 63), (100, 50), 0, 0, 180, (255, 0, 0), -1]
    
    # The code below is creating a Python script that uses the `st` module to create a user interface
    # for drawing an ellipse. It creates a main container with a border, adds a subheader for the
    # "Drawing Ellipse" section, and provides a markdown explanation of how to draw an ellipse using
    # the `cv.ellipse()` function. It also creates an image container and a code container within the
    # main container.
    
    main_container = st.container(border=True)
    main_container.subheader("Drawing Ellipse")
    main_container.markdown("""
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
    
    image_container = main_container.container(border=True)
    code_container = main_container.container(border=True)
    
    with st.sidebar:
        
        st.markdown("<center><h5>Parameters</h5></center>", unsafe_allow_html=True)
        st.info("Change Parameters to see differences")
        
        with st.container(border=True):
        
            st.markdown("<center>Center</center>",
                        unsafe_allow_html=True)
            center = st.slider("`x - coordinate`",value=447,max_value=512),\
                                st.slider("` y - coordinate`", value=63, max_value= 512)

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
            
            thickness=st.slider("Thickness",value=-1, min_value=-1,
                                        max_value=10, label_visibility="hidden")
            
            # The code is checking the value of the variable `thickness`. If `thickness` is less than
            # 0, it assigns `True` to the variable `val`. Otherwise, it assigns `False` to `val`.
            val = True if thickness<0 else False
            
            if st.checkbox("Fill? ", value=val): thickness=-1
    
    image_container.subheader("Output")
    code_container.subheader("Code")
   
    # The code below  is checking if the parameters for drawing an ellipse (center, axes_length, angle,
    # start_angle, end_angle, color, thickness) are equal to the default values. If they are not equal
    # to the defaults, it will draw an ellipse with the given parameters and display the output image.
    # It will also display the code used to draw the ellipse. If the parameters are equal to the
    # defaults, it will draw an example ellipse with the default parameters and display the example
    # output image. It will also display the example code used to draw the ellipse.
    
    if [center, 
        axes_length,
        angle,
        start_angle,
        end_angle,
        color,
        thickness] != defaults:
        
        image_container.image(draw_ellipse( center, 
                            axes_length,
                            angle,
                            start_angle,
                            end_angle,
                            color,
                            thickness ),
                'Draw Ellipse',
                use_column_width=True)
        image_container.success("Your Output")
        
        code_container.success("Your Code")
        code_container.code(write_code(center, 
                            axes_length,
                            angle,
                            start_angle,
                            end_angle,
                            color,
                            thickness))
        
    else:
        image_container.image(draw_ellipse(*defaults),
        'Draw Ellipse',
        use_column_width=True)
        image_container.info("Example Output")
        
        code_container.info("Example Code")
        code_container.code(write_code(center, 
                            axes_length,
                            angle,
                            start_angle,
                            end_angle,
                            color,
                            thickness))

def Draw_Polygon():
    
    def write_code(pts, join, color):
        return f"""
                import numpy as np
                import cv2 as cv
                # Create a black image
                img = np.zeros((512,512,3), np.uint8)
                pts = np.array({pts}, np.int32)
                pts = pts.reshape((-1,1,2))
                cv.polylines(img,[pts], {join}, {color})
                cv.imshow('Output', img)
                """
    
    defaults = [[[10,5],[20,30],[70,20],[50,10]], True, (255,255,255)]
    
    main_container = st.empty().container(border=True)
    
    main_container.subheader("Drawing Polygons")
    main_container.markdown("""
                To draw a polygon, first you need coordinates of vertices.
                Make those points into an array of shape ROWSx1x2 where ROWS
                are number of vertices and it should be of type int32. 
                Here we draw a small polygon of with four vertices in yellow color.
                """)
    
    image_container = main_container.container(border=True)
    code_container = main_container.container(border=True)

    main_container.markdown("""
                #### Note
                > If third argument is False, you will get a polylines
                joining all the points, not a closed shape.
                `cv.polylines()` can be used to draw multiple lines. 
                Just create a list of all the lines you want to draw 
                and pass it to the function.
                All lines will be drawn individually. 
                It is a much better and faster way to draw a group of lines
                than calling cv.line() for each line.
                """)
    
    with st.sidebar:
        st.markdown("<center> <h5> Parameters </h5> </center>", unsafe_allow_html=True)
        st.info("Change Parameters to see differences")
        points = st.sidebar.number_input("How many points for your Polygon?", value=0)
        pts = []
        with st.container(border=True):
            if points:
                for i in range(points):
                    st.markdown(f"Coordinates for point : `{i+1}`")
                    x = st.slider(f"$x_{i+1}$:", key=f"x{i}")
                    y = st.slider(f"$y_{i+1}$:", key=f"y{i}")
                    pts.append([x, y])
                    
            join = st.checkbox("Join?", value=True)
            color = st.color_picker("Color:", value="#00ffff")
            color = ImageColor.getcolor(f'{color}','RGB')
        
    image_container.subheader("Output")
    code_container.subheader("Code")
    
    if points: 
        image_container.image(draw_polygon(pts, join, color), caption="Drawing Polygon", use_column_width=True)
        image_container.success("Your Output")
        code_container.code(write_code(pts, join, color))
        code_container.success("Your Code")
    else:
        image_container.image(draw_polygon(), caption="Drawing Polygon", use_column_width=True)
        image_container.info("Example Output")
        code_container.code(write_code(*defaults))
        code_container.info("Example Code")
        

def Draw_Text():
    
    def write_code(text='OpenCV',
              position=(10, 500),
              font='HERSHEY_SIMPLEX',
              font_scale=4,
              color = (255,255,255),
              thickness=2,
              lineType='LINE_AA'):
        
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
                        {font_scale}, {color[::-1]}, {thickness}, {line_type_dict[lineType]})
            cv.imshow('Output', img)
            """
                
    with st.container(border=True):
        st.subheader("Adding Text to Images")
        st.markdown("""
                    To put texts in images, you need specify following things.

                    - Text data that you want to write
                    - Position coordinates of where you want put it 
                    (i.e. bottom-left corner where data starts).
                    - Font type (Check cv.putText() docs for supported fonts)
                    - Font Scale (specifies the size of font)
                    - regular things like color, thickness, lineType etc. 
                    For better look, lineType = cv.LINE_AA is recommended.
                    
                    We will write OpenCV on our image in white color.""")
                
        image_container=st.empty()
        code_container = st.empty().container(border=True)
        
    st.sidebar.info("Change Parameters to see differences")
    
    st.sidebar.subheader("Parameters")
    
    with st.sidebar.container(border=True):
        
        text=st.text_input("Add text", placeholder="Add some text...")
        text_info=st.empty()
        
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
        
        live = False
        add_text=st.button("Add Text", type="primary", use_container_width=True)
        
        if not text:
            text_info.error('Please Add text to see difference')
        else:
            live=True
            
    with image_container.container(border=True):
        st.markdown("<center> Output </center>", unsafe_allow_html=True)
        if add_text or live: 
            st.image(draw_text(text, position, font, font_scale, color, thickness, lineType),
                     caption="Adding Texts to Images", use_column_width=True)
            st.success("Your output")
            code_container.markdown("### Code")
            code_container.success("Your modified code")
            code_container.code(write_code(text, position, font, font_scale, color, thickness, lineType))
        else:
            st.image(draw_text(), caption="Adding Texts to Images", use_column_width=True)
            st.info("Example output")
            code_container.markdown("### Code")
            code_container.info("Example code")
            code_container.code(write_code())