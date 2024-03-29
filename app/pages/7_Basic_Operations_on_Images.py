import streamlit as st
from utils.gui import menu, footer, images
 
def main():
    st.markdown("# Basic Operations on Images")
    basicOp= images.BasicOperations()
    with st.container(border=True):
        st.subheader("Topics")
        options=st.radio("Select: ", ["Introduction", 
                                              "Accessing and Modifying pixel values",
                                              "Accessing Image Properties",
                                              "Image ROI",
                                              "Splitting and Merging Image Channels",
                                              "Making Borders for Images (Padding)"], 
                         horizontal=True,
                         label_visibility="collapsed")

    
    if options == "Introduction":
        st.markdown("""
                    ## Goal
                    **Learn to:**
                            
                    - Access pixel values and modify them
                    - Access image properties
                    - Set a Region of Interest (ROI)
                    - Split and merge images

                    Almost all the operations in this section are mainly related to 
                    [Numpy]('https://numpy.org/') rather than OpenCV. 
                    """)
        st.info("""
                **Note:** \n > A good knowledge of Numpy is required 
                to write better optimized code with OpenCV.""")
        st.markdown ("""*( Examples will be shown in a Python terminal,
                 since most of them are just single lines of code )*""")
    
    if options == "Accessing and Modifying pixel values":
            basicOp.Accessing_Modifying_Pixel_Values()
        
    if options == "Accessing Image Properties":
        basicOp.Accessing_Image_Properties()
    
    if options == "Image ROI":
        basicOp.Image_ROI()
    
    if options == "Splitting and Merging Image Channels":
        basicOp.Splitting_and_Merging_Image_Channels()
        
    if options == "Making Borders for Images (Padding)":
        basicOp.Making_Borders_for_Images()
    
if __name__ == '__main__':
    st.set_page_config(page_icon="app/assets/Images/OpenCV_Logo_with_text.png",
                       page_title="Basic Operations on Images")
    
    menu.menu()
    main()
    footer.footer()