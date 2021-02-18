import streamlit as st
from PIL import Image
import pickle
import PyPDF2
from pipeline.preprocess.preprocess import Preprocess

model = pickle.load(open('pipeline/model/clf.pickle', 'rb'))

def main():
    st.title("HumAIn AP")
    image = Image.open("assets/research.png")
    st.image(image, caption=None, width=100)
    st.markdown("###")
    st.header("Upload your document to be analyzed")
    filename = st.file_uploader("Upload", type=["pdf", "txt"])
    if filename is not None:
        if st.button("Analyze"):
            file_location = "assets/uploads/" + filename.name
            with open(file_location, "wb") as f:
                f.write(filename.getbuffer())
                tis = ["Hello", "World"]
                st.write(tis[0])
            extracted_text = pdf_file_read(file_location)
            st.write(type(extracted_text))
            nlp_model = Preprocess()
            st.write(nlp_model.preprocess(extracted_text))


def pdf_file_read(file_name):
    # creating a pdf file object
    pdfFileObj = open(file_name, 'rb')  
        
    # creating a pdf reader object  
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)  
        
    # printing number of pages in pdf file  
    print(pdfReader.numPages)  
        
    # creating a page object  
    pageObj = pdfReader.getPage(0)  
        
    # extracting text from page  
    text = pageObj.extractText() 
        
    # closing the pdf file object  
    pdfFileObj.close()

    return text


# def extract_data()


if __name__ == "__main__":
    main()