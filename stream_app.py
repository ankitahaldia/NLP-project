import streamlit as st
from PIL import Image

def main():
    st.title("HumAIn AP")
    image = Image.open("assets/research.png")
    st.image(image, caption=None, width=100)
    st.markdown("###")
    st.header("Upload your document to be analyzed")
    filename = st.file_uploader("Upload", type=["pdf", "txt"])
    if filename is not None:
        st.write(filename)
    st.button("Analyze")


if __name__ == "__main__":
    main()