import streamlit as st
from PIL import Image
import pickle

model = pickle.load(open('pipeline/model/clf.pickle', 'rb'))

def main():
    st.title("HumAIn AP")
    image = Image.open("assets/research.png")
    st.image(image, caption=None, width=100)
    st.markdown("###")
    st.header("Upload your document to be analyzed")
    filename = st.file_uploader("Upload", type=["pdf", "txt", "jpg"])
    if filename is not None:
        if st.button("Analyze"):
            tis = ["Hello", "World"]
            st.write(tis[0])
    


if __name__ == "__main__":
    main()