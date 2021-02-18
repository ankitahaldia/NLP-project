
import nlp_preprocess

#testing
from os import getcwd as cwd
from os.path import dirname as dir
from os.path import join
import pandas as pd


# Multiple labels classification
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import LabelPowerset

# Single label classification
import spacy
from spacy import load
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.preprocessing import MultiLabelBinarizer

#model saving
from pickle import dump

#typing
from typing import List
from typing import Dict

INDUSTRIES: Dict[str, List[str]] = {
    'automative': ['automotive', 'taxi', 'wheel', 'fuel', 'car', 'drive', 'auto', 'selfdrive', 'vehicle', 'road',
                   'automobile'],
    'Manufacturing': ['cleantech', 'deindustrialization', 'prefabrication', 'manufacturing', 'vitrification',
                      'fabrication' 'R&D', 'quality', 'produce', 'goods', 'factory', 'equipment'],
    'Consumer Products': ['product', 'price', 'goods', 'commerce', 'economic', 'customer', 'marketing', 'demand',
                          'inventory', 'supply'],
    'Finance': ['bank', 'money', 'capitalization', 'interest', 'fund', 'finance', 'asset', 'risk', 'loan', 'credit',
                'fraud'],
    'Agriculture': ['soil', 'grain', 'agriculture', 'field', 'farm', 'soil', 'weather', 'crop', 'grow', 'animal',
                    'food', 'land'],
    'Energy': ['renewable', 'sustainable', 'green', 'electricity', 'energy', 'power', 'mines', 'solar', 'light',
               'metal', 'electric', 'carbon', 'electonic', 'wind', 'speed'],
    'Health Care': ['Health', 'Ca re', 'emergency', 'doctor', 'wellness', 'patient', 'hospital', 'clinic', 'treatment',
                    'disease', 'medical', 'cancer'],
    'Pharmaceuticals': ['dose', 'pillbox', 'tonic', 'tablet', 'placebo', 'medicate', 'hospital', 'Pharmaceutical',
                        'drug', 'diagnose', 'test', 'trial', 'medicine', 'vaccine'],
    'Public and Social sector': ['social', 'law', 'crime', 'terrorism', 'policing', 'govern', 'public',
                                 'infrastructure', 'education', 'tax', 'urban', 'life', 'job', 'enforcement',
                                 'surveillance'],
    'Media': ['mainstream', 'publishing', 'medium', 'social', 'media', 'video', 'content', 'news', 'release', 'film',
              'press', 'viral', 'game'],
    'Telecom': ['location', 'station', 'host', 'telecom', 'mobile', 'voice', 'call', 'subscription', 'network', 'phone',
                'broadcast', 'internet', 'communication', 'modulation'],
    'Transport & Logistics': ['transport', 'logistic', 'mail', 'parcel', 'travel', 'route', 'planes', 'truck',
                              'shipping', 'mobility', 'movement']}



class Model:
    def __init__(self, nlp_model='en_core_web_md'):
        self.nlp = load(nlp_model)
        self.groups_keywords = {}
        self.groups_keywords['industry'] = INDUSTRIES
        self.Preprocess = nlp_preprocess.Preprocess(nlp_model='en_core_web_md')

    def train(self, data: DataFrame, X_column: str, y_columns: List[str] = None):
        if y_columns is None:
            _ = data.columns.to_list()
            y_columns = list(set(_)-set([X_column]))
        X = data[X_column]
        y: DataFrame = data.drop(X_column, axis=1)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.2)
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(ytrain[y_columns].values)
        # test_labels not used when training
        # test_labels = mlb.fit_transform(ytest[y_columns].values)
        train_cleaned = xtrain.copy(deep=True).apply(nlp_preprocess.Preprocess().clean_text)
        # test cleaned not used when training
        # test_cleaned = xtest.copy(deep=True).apply(clean_text)
        vectorizer = TfidfVectorizer()
        vectorised_train_documents = vectorizer.fit_transform(train_cleaned)
        powersetsvc = LabelPowerset(LinearSVC())
        powersetsvc.fit(vectorised_train_documents, train_labels)
        dump(powersetsvc, open("powersetsvc.pickle", "wb"))
        with open('vec.pickle', 'wb') as f1:
            dump(vectorizer, f1)
        return powersetsvc, vectorizer


    def predict_group(self, texts_tokens: List[List[str]], groups_keywords: Dict[str, List[str]] = None,
                      identifiers_number=20) -> List:
        if groups_keywords is None:
            groups_keywords = self.groups_keywords['industry']
        for keys, value in groups_keywords.items():
            groups_keywords[keys] = " ".join(value)
        # Create a Dictionary from the articles: dictionary
        dictionary = Dictionary(texts_tokens)
        # Create a MmCorpus
        corpus = [dictionary.doc2bow(tt) for tt in texts_tokens]
        # Create a new TfidfModel
        tfidf = TfidfModel(corpus)
        tfidf_weights = [sorted(tfidf[doc], key=lambda w: w[1], reverse=True) for doc in corpus]
        # Create nlp object for each group and keywords
        key_list = list(groups_keywords.keys())
        groups_keywords_docs = []
        for key, value in groups_keywords.items():
            groups_keywords_docs.append(self.nlp(groups_keywords[key]))
        # Predict industry type with similarity method along two nlp object for all texts in industry_type_list
        # Select the most frequent words for each text a and convert list into string
        texts_identifiers = [[dictionary.get(term_id) for term_id, weight in tfidf_weight[:identifiers_number]]
                             for tfidf_weight in tfidf_weights]
        texts_identifiers = [" ".join(text_identifiers) for text_identifiers in texts_identifiers]
        groups = []
        for ti in texts_identifiers:
            # Create nlp object based only on text identifiers
            doc_publication = self.nlp(ti)
            similarities = []
            for gkd in groups_keywords_docs:
                similarities.append(doc_publication.similarity(gkd))
            max_value = max(similarities)
            max_position = similarities.index(max_value)
            industry_type = key_list[max_position]
            groups.append(industry_type)
        return groups

#testing
if str(cwd()).find('belearner'):
    START_PATH = dir(cwd())
    DATA_PATH = join(START_PATH + r'\data\new_file.csv')
    #testing a 100 docs sample
    df = pd.read_csv(DATA_PATH, delimiter='\t').sample(10)
    model = Model(nlp_model='en_core_web_sm')
    powersetsvc, vectorizer = model.train(data= df,X_column= 'text')


