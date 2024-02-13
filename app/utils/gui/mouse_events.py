import streamlit as st
import cv2 as cv


def Simple_Demo():
    st.markdown("""<h2> Simple Demo<h2>
                        <p> 
                        Here, we create a simple application which draws a circle on an image wherever we double-click on it. <br>
                            First we create a mouse callback function which is executed when a mouse event take place. Mouse event can be anything related to mouse like left-button down,
                            left-button up, left-button double-click etc. 
                            It gives us the coordinates <code>(x,y)</code> for every mouse event. 
                            With this event and location, we can do whatever we like. 
                            To list all available events available, run the following code in Python terminal:
                        </p>
    """, unsafe_allow_html=True)
        
    st.code("""
                import cv2 as cv
                events = [i for i in dir(cv) if 'EVENT' in i]
                print( events )
    """)
        
    events = [i for i in dir(cv) if 'EVENT' in i]
    
    with st.expander("Output: "):
        st.write(events)
    
    st.markdown("""
                Creating mouse callback function has a specific format 
                which is same everywhere. 
                It differs only in what the function does. 
                So our mouse callback function does one thing,
                it draws a circle where we double-click. 
                So see the code below. 
                Code is self-explanatory from comments :
                """)
    st.code("""
            import numpy as np
import cv2 as cv
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,0,0),-1)
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
            """)
    with st.expander("Output:"):
        st.image("app/assets/GIFs/mouse_events.gif", use_column_width=True)

def Advanced_Demo():
    pass
