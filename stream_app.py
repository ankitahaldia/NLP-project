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
    filename = st.file_uploader("Upload", type=["pdf", "txt"])
    if filename is not None:
        st.write(model.predict("Cybersecurity provider ZingBox has announced the launch of a new generation of cybersecurity solutions, called IoT Guardian, which focuses on service protection."))
        st.write(filename)
    st.button("Analyze")


if __name__ == "__main__":
    main()