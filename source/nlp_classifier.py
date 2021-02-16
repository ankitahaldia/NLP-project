from pandas import Series

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# import nlp to create Doc objects
import nlp

from spacy import load, lang

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from wordcloud import WordCloud

from collections import Counter

from typing import List
from typing import Dict

# GLOBAL VARIABLES
NES = ['microsoft']
INDUSTRIES = Dict[List[str]] = {
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
    'Health Care': ['Health', 'Care', 'emergency', 'doctor', 'wellness', 'patient', 'hospital', 'clinic', 'treatment',
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
                              'shipping', 'mobility', 'movement']
}


def lemmatize(texts_tokens: List[List[str]]) -> List[List[str]]:
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    # Lemmatize all tokens into a new list: lemmatized
    texts_lemmatized = [[wordnet_lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in text_tokens]
                        for text_tokens in texts_tokens]
    return texts_lemmatized


def filter_words(texts_lemmatized: List[List[str]], freq_min=None, freq_max=None) -> List[List[str]]:
    frequency_absolute = Counter([item for elem in texts_lemmatized for item in elem])
    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(frequency_absolute)
    frequency_relative = wordcloud.words_
    if freq_min is not None and freq_min > 0 and freq_min < 1:
        rel_freq_filtered = {k: v for k, v in frequency_relative.items() if v > freq_min}
    if freq_max is not None and freq_max > 0 and freq_max < 1:
        rel_freq_filtered = {k: v for k, v in frequency_relative.items() if v < freq_max}
    texts_filtered = [[t for t in pub_lem if t in rel_freq_filtered.keys()] for pub_lem in texts_lemmatized]
    return texts_filtered


class Nlp_Classifier:
    def __init__(self, publications_text: Series, nlp_model='en_core_web_md'):
        self.nlp = load(nlp_model)
        self.stopwords = lang.en.stop_words.STOP_WORDS
        self.nes = set(NES)
        self.groups_keywords = {}
        self.groups_keywords['industry'] = INDUSTRIES

    def get_named_entities(self, texts: Series, inplace=True) -> set:
        # creating a single ner set
        nes = set()

        # function to extract NER from text
        def get_named_entities(text) -> set:
            doc = self.nlp(text)
            named_entities = set([ent.text for ent in doc.ents])
            return named_entities

        [[nes.add(n) for n in get_named_entities(text)] for text in texts]
        # adding predefined NER
        [nes.add(n) for n in self.NER]
        if inplace:
            self.nes = nes
        return nes

    def tokenize(self, texts: Series) -> List[List[str]]:
        # Tokenize the article: tokens
        tokens = [word_tokenize(text) for text in texts]
        # Convert the tokens into lowercase: lower_tokens
        lower_tokens = [[t.lower() for t in token] for token in tokens]
        # Retain alphabetic words: alpha_only
        alphas = [[t for t in lower_token if t.isalpha()] for lower_token in lower_tokens]
        # Remove all stop words: no_stops
        no_stops = [[t for t in alpha if t not in self.stopwords] for alpha in alphas]
        # Remove all NER
        no_nes = [[t for t in no_stop if t not in self.nes] for no_stop in no_stops]
        return no_nes

    def predict_group(self, texts_tokens: List[List[str]], groups_keywords: Dict[List[str]] = None,
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
            groups_keywords_docs.append(nlp(groups_keywords[key]))
        # Predict industry type with similarity method along two nlp object for all texts in industry_type_list
        # Select the most frequent words for each text a and convert list into string
        texts_identifiers = [[dictionary.get(term_id) for term_id, weight in tfidf_weight[:identifiers_number]]
                             for tfidf_weight in tfidf_weights]
        texts_identifiers = [" ".join(text_identifiers) for text_identifiers in texts_identifiers]
        groups = []
        for ti in texts_identifiers:
            # Create nlp object based only on text identifiers
            doc_publication = nlp(ti)
            similarities = []
            for gkd in groups_keywords_docs:
                similarities.append(doc_publication.similarity(gkd))
            max_value = max(similarities)
            max_position = similarities.index(max_value)
            industry_type = key_list[max_position]
            groups.append(industry_type)
        return groups