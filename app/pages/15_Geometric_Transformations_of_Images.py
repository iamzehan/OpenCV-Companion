import streamlit as st 
from utils.gui import footer, menu, images

def main():
    st.title("""
             Geometric Transformations on Images
             """)
    
    geoTrans = images.GeometricTransformations()
    
    with st.container(border=True):
        st.subheader("Topics")
        options = st.radio("Topics", options=['Introduction',
                                    'Scaling',
                                    'Translation',
                                    'Rotation',
                                    'Affine Transformation',
                                    'Perspective Transformation'],
                           horizontal=True,
                           label_visibility='collapsed')
        
    if options == 'Introduction':
        st.subheader("Goals")
        st.markdown("""
            Learn to apply different geometric transformations to images, such as translation, rotation, affine transformation, etc.
            You will encounter these functions: `cv2.getPerspectiveTransform`
            ### Transformations
            OpenCV provides two transformation functions, 
            `cv2.warpAffine` and `cv2.warpPerspective`, with which you can perform all kinds of transformations. 
            `cv2.warpAffine` takes a 2x3 transformation matrix, while `cv2.warpPerspective` takes a 
            3x3 transformation matrix as input.
            """)

    elif options == 'Scaling':
        st.markdown("## Scaling")
        geoTrans.Scaling()

    elif options == 'Translation':
        st.markdown("## Translation")
        geoTrans.Translation()

    elif options == 'Rotation':
        st.markdown("## Rotation")
        geoTrans.Rotation()

    elif options == 'Affine Transformation':
        st.markdown("## Affine Transformation")
        geoTrans.AffineTransformation()
        
    elif options == 'Perspective Transformation':
        st.markdown("## Perspective Transformation")
        geoTrans.PerspectiveTransform()

if __name__ == '__main__':
    st.set_page_config("Geometric Transformations of Images", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()

    main()
    footer.footer()