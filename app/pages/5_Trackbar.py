import streamlit as st
import numpy as np
from utils.gui import menu, footer

def trackbar(r, g, b, s):
    img = np.zeros((300,512,3), np.uint8)
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
    return img

def main():
        
    st.title("Goal")
    st.markdown("""
            - Learn to bind trackbar to OpenCV windows
            - You will learn these functions : `cv.getTrackbarPos()`, `cv.createTrackbar()` etc.
""")
    st.markdown("""
                ## Code Demo
                Here we will create a simple application which shows the color you specify.
                You have a window which shows the color and three trackbars to specify each of B,G,R colors.
                You slide the trackbar and correspondingly window color changes. By default, initial color will be set to Black.
                For `cv.createTrackbar()` function, first argument is the trackbar name, second one is the window name to which it is attached,
                third argument is the default value, 
                fourth one is the maximum value and fifth one is the callback function which is executed every time trackbar value changes.
                The callback function always has a default argument which is the trackbar position. In our case, function does nothing, so we simply pass.
                Another important application of trackbar is to use it as a button or switch.
                OpenCV, by default, doesn't have button functionality. So you can use trackbar to get such functionality.
                In our application, we have created one switch in which application works only if switch is `ON`, otherwise screen is always black.
            """)
    st.code("""
            import numpy as np
            import cv2 as cv
            def nothing(x):
                pass
            # Create a black image, a window
            img = np.zeros((300,512,3), np.uint8)
            cv.namedWindow('image')
            # create trackbars for color change
            cv.createTrackbar('R','image',0,255,nothing)
            cv.createTrackbar('G','image',0,255,nothing)
            cv.createTrackbar('B','image',0,255,nothing)
            # create switch for ON/OFF functionality
            switch = '0 : OFF \\n1 : ON'
            cv.createTrackbar(switch, 'image',0,1,nothing)
            while(1):
                cv.imshow('image',img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break
                # get current positions of four trackbars
                r = cv.getTrackbarPos('R','image')
                g = cv.getTrackbarPos('G','image')
                b = cv.getTrackbarPos('B','image')
                s = cv.getTrackbarPos(switch,'image')
                if s == 0:
                    img[:] = 0
                else:
                    img[:] = [b,g,r]
            cv.destroyAllWindows()
            """)
    with st.expander("Output"):
        st.write("The screenshot of the application looks like below :")
        st.image("app/assets/Images/trackbar_screenshot.png",use_column_width=True)
    
    with st.container(border=True):
        st.subheader("Playground")
        r = st.slider('R', value = 255, min_value=0, max_value=255)
        g = st.slider('G', value = 172, min_value=0, max_value=255)
        b = st.slider('B', value = 31, min_value=0, max_value=255)

        s = st.slider('0 : OFF \n1 : ON', value = 1, min_value=0, max_value=1)

        st.image(trackbar(r, g, b, s), channels='BGR', caption="image", use_column_width=True)

if __name__ == '__main__':
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Track Bar 📊")
    menu.menu()
    main()
    footer.footer()