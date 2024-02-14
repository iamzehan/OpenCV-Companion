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
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Track Bar 📊")
    
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
            # File and name handling
            img_file_name = 'messi5.jpg'
            st.markdown("""## Accessing and Modifying pixel values
Let's load a color image first:
                """)
            render = st.empty().container()
            render.subheader("Code")
            st.sidebar.info("Upload an image to see changes")
            img_file = st.sidebar.file_uploader("Upload an Image to see how the code changes:", type=["PNG","JPG"], label_visibility="collapsed")
            
            # Checks if a File has been uploaded
            if img_file:
                # extracting name img_file object of the Upload class
                img_file_name = img_file.name
                # rendition of the whole view
                Accessing_Modifying_Pixel_Values(img_file, img_file_name, render, upload=True)
            else:
                Accessing_Modifying_Pixel_Values(img_file, img_file_name, render)
        
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