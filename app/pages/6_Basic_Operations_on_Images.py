import streamlit as st
from utils.gui.footer import footer
from utils.gui.images import \
    (
        Accessing_Modifying_Pixel_Values,
        Accessing_Image_Properties,
        Image_ROI,
        Splitting_and_Merging_Image_Channels,
        Making_Borders_for_Images
    )
    
def main():

    options=st.sidebar.selectbox("Select: ", ["Introduction", 
                                              "Accessing and Modifying pixel values",
                                              "Accessing Image Properties",
                                              "Image ROI",
                                              "Splitting and Merging Image Channels",
                                              "Making Borders for Images (Padding)"], label_visibility="collapsed")
    
    if options == "Introduction":
        st.title("Goal")
        st.markdown("""

**Learn to:**
           
- Access pixel values and modify them
- Access image properties
- Set a Region of Interest (ROI)
- Split and merge images

Almost all the operations in this section are mainly related to 
[Numpy]('https://numpy.org/') rather than OpenCV. 

""")
        st.info("""**Note:** \n > A good knowledge of Numpy is required 
                to write better optimized code with OpenCV.""")
        st.markdown ("""*( Examples will be shown in a Python terminal,
                 since most of them are just single lines of code )*""")
        
    if options == "Accessing and Modifying pixel values":
        
        Accessing_Modifying_Pixel_Values()
    
    if options == "Accessing Image Properties":
        
        Accessing_Image_Properties()
    
    if options == "Image ROI":
        
        Image_ROI()
    
    if options == "Splitting and Merging Image Channels":
        
        Splitting_and_Merging_Image_Channels()
        
    if options == "Making Borders for Images (Padding)":
        
        Making_Borders_for_Images()
    
if __name__ == '__main__':
    main()
    footer()