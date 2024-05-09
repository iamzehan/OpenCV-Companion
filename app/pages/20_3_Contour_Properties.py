import streamlit as st
from utils.gui import footer, menu, images
def main():
    properties = images.Contours.Properties()
    st.title("Contour Properties")
    st.subheader("Goals")
    st.markdown("""
             Here we will learn to extract some frequently used properties 
             of objects like Solidity, Equivalent Diameter, Mask image, Mean Intensity etc. 
             More features can be found at [Matlab regionprops documentation](http://www.mathworks.in/help/images/ref/regionprops.html).
             """)
    
    functions = {
        "Aspect Ratio": properties.AspectRatio,
        "Extent": properties.Extent,
        "Solidity": properties.Solidity,
        "Equivalent Diameter": properties.EquivalentDiameter,
        "Orientation": properties.Orientation,
        "Mask and Pixel Points": properties.Mask_and_Pixel_Points,
        "Maximum Value, Minimum Value and their locations": properties.MaxVal_MinVal_locations,
        "Maximum Color or Mean Intensity": properties.MaximumColor_or_MeanIntensity,
        "Extreme Points": properties.ExtremePoints
    }
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio(label="Options: ",
                        options=list(functions.keys()),
                        horizontal = True,
                        label_visibility="collapsed")
    if options:
        functions[options]()
            
if __name__ == "__main__":
    st.set_page_config("Contours in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
