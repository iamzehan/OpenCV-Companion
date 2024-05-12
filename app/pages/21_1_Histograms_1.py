import streamlit as st
from utils.gui import footer, menu, images
def main():
    st.title("Histograms - 1 : Find, Plot, Analyze !!!")
    st.markdown("Learn to find and draw Contours")
    
    hist1 = images.Histograms.Histograms1()
    functions = {
        "Introduction" : hist1.Introduction,
        "Find Histogram" : hist1.FindHistogram,
        "Plotting Histogram": hist1.PlottingHistograms,
        "Application of Mask": hist1.ApplicationOfMask 
    }
    with st.container(border=True):
                st.subheader("Topics")
                options = st.radio(label="Options: ",
                                options=list(functions.keys()),
                                horizontal = True,
                                label_visibility="collapsed")
            
    if options: functions[options]()
        
if __name__ == "__main__":
    st.set_page_config("Histograms in OpenCV", page_icon='app/assets/Images/OpenCV_Logo_with_text.png')
    menu.menu()
    main()
    footer.footer()
