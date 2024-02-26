import streamlit as st
from utils.gui.footer import footer
from  utils.gui.menu import core_operations_menu

def main():
    st.markdown("""
                <center> 
                    <h2> Performance Measurement and Improvement Techniques 
                    </h2> 
                </center>
                
                """, unsafe_allow_html=True)
    
    options = st.sidebar.selectbox("Select: ", options=['Introduction',
                                                'Measuring Performance with OpenCV',
                                                'Default Optimization',
                                                'Measuring Performance in IPython',
                                                'More IPython magic Commands',
                                                'Performance Optimization Techniques'],
                           label_visibility="collapsed")

    if options == "Introduction":
        st.markdown("""
                    ### Goal
                    In image processing, since you are dealing with large
                    number of operations per second, it is mandatory that 
                    your code is not only providing the correct solution, 
                    but also in the fastest manner. So in this chapter, 
                    you will learn

                    - To measure the performance of your code.
                    - Some tips to improve the performance of your code.
                    - You will see these functions : `cv2.getTickCount`, 
                    `cv2.getTickFrequency` etc.
                    
                    Apart from OpenCV, Python also provides a module time 
                    which is helpful in measuring the time of execution. 
                    Another module profile helps to get detailed report 
                    on the code, like how much time each function in the 
                    code took, how many times the function was called etc.
                    But, if you are using IPython, all these features are
                    integrated in an user-friendly manner. We will see some
                    important ones, and for more details, check links in 
                    Additional Resouces section.
                    """)

    elif options == "Measuring Performance with OpenCV":
        st.subheader("Measuring Performance with OpenCV")
       
    elif options == "Default Optimization":
        st.subheader("Default Optimization")
        
    elif options == "Measuring Performance in IPython":
        st.subheader("Measuring Performance in IPython")
        
    elif options == "More IPython magic Commands":
        st.subheader("More IPython magic Commands")
        
    elif options == "Performance Optimization Techniques":
        st.subheader("Performance Optimization Techniques")
        
if __name__ == '__main__':
    st.set_page_config(page_title="Performance Measurement & Improvement",
                       page_icon="app/assets/Images/OpenCV_Logo_with_text.png")
    core_operations_menu()
    main()
    footer()