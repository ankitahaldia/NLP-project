import streamlit as st
from PIL import Image
import pickle
import PyPDF2
import nltk
import numpy as np
from scipy.sparse import lil_matrix
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS
from nlp_preprocess import Preprocess

model = pickle.load(open('pipeline/model/clf.pickle', 'rb'))
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
english_stops = set(STOPWORDS)

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
            # st.write(type(extracted_text))
            # st.write(prediction(extracted_text))
            nlp_model = Preprocess()
            st.write(nlp_model.clean_text(extracted_text))


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


def prediction(extracted_text) :
    
    with open("pipeline/model/clf.pickle", 'rb') as model_file :
        model = pickle.load(model_file)
    with open("pipeline/model/vec.pickle", 'rb') as vec_file :
        vectorizer = pickle.load(vec_file)

    text_processed = tokenize_lemma_stopwords(extracted_text)
    new_corpus = [text_processed]
    new_X_test = vectorizer.transform(new_corpus).toarray()

    pred = model.predict(new_X_test)
    pred1 = lil_matrix.toarray(pred)

    industry_dict = {'Agriculture': 4,
        'Consumer Products': 2,
        'Energy': 5,
        'Finance': 3,
        'Health Care': 6,
        'Manufacturing': 1,
        'Media': 9,
        'Pharmaceuticals': 7,
        'Public and Social sector': 8,
        'Telecom': 10,
        'Transport & Logistics': 11,
        'automative': 0}
    key_list = list(industry_dict.keys())
    positions = np.where(pred1 == 1)[1]
    predicted_industries = []
    predicted_industries.append(key_list[positions[0]])
    predicted_industries.append(key_list[positions[1]])

    return predicted_industries


def tokenize_lemma_stopwords(text):
    text = text.replace("\n", " ")
    tokens = nltk.tokenize.word_tokenize(text.lower()) # split string into words (tokens)
    tokens = [t for t in tokens if t.isalpha()] # keep strings with only alphabets
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [t for t in tokens if t not in english_stops] # remove stopwords
    cleanedText = " ".join(tokens)
    return cleanedText


if __name__ == "__main__":
    main()