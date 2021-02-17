import streamlit as st
from PIL import Image

def main():
    st.title("HumAIn AP")
    image = Image.open("assets/research.png")
    st.image(image, caption=None, width=100, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.markdown("###")
    st.write("Upload your document to be analyzed")
    filename = st.file_uploader("Upload", type=["pdf", "txt"])
    if filename is not None:
        st.write(type(filename))


if __name__ == "__main__":
    main()