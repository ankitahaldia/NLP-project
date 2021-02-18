from pandas import Series

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
# import entirely spacy to create Doc objects through nlp
import spacy
from spacy import load, lang

from wordcloud import WordCloud

from collections import Counter

from typing import List
from typing import Dict

# WARNINGS
# W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
# I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine

# GLOBAL VARIABLES
NAMED_ENTITIES = ['microsoft']

#testing
from os import getcwd as cwd
from os.path import dirname as dir
from os.path import join
import pandas as pd


def lemmatize(text_tokens: List[str]) -> List[str]:
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
    texts_lemmatized = [wordnet_lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in text_tokens]
    return texts_lemmatized


#DEV NOTE: not used
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


class Preprocess:
    def __init__(self, nlp_model='en_core_web_md'):
        self.nlp = spacy.load(nlp_model)
        self.stop_words = lang.en.stop_words.STOP_WORDS
        self.named_entities = set(NAMED_ENTITIES)

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
        [nes.add(n) for n in self.named_entities]
        if inplace:
            self.named_entities = nes
        return nes

    def tokenize_text(self, text:str, stop_words: List[str] = None, named_entities: List[str] = None,
                   lenght_min: int=2) -> List[str]:
        if stop_words is None:
            stop_words = self.stop_words
        if named_entities is None:
            named_entities = self.named_entities
        text = text.replace("\n", " ")
        # split string into words (tokens)
        tokens = word_tokenize(text.lower())
        # keep strings with only alphabets
        tokens = [t for t in tokens if t.isalpha()]
        tokens = lemmatize(tokens)
        # remove short words, they're probably not useful
        tokens = [t for t in tokens if len(t) > lenght_min]
        # remove stopwords
        tokens = [t for t in tokens if t not in stop_words]
        # remove
        tokens = [t for t in tokens if t not in named_entities]
        return tokens

    def clean_text(self, text:str, stop_words: List[str] = None, named_entities: List[str] = None,
                   lenght_min: int=2) -> str:
        tokens = self.tokenize_text(text, stop_words, named_entities, lenght_min)
        text_cleaned = " ".join(tokens)
        return text_cleaned

    def tokenize_texts(self, texts:Series, stop_words: List[str] = None, named_entities: List[str] = None,
                   lenght_min: int=2) -> List[List[str]]:
        texts_tokens = []
        for text in texts:
            texts_tokens = texts_tokens.append(self.tokenize_text(text, stop_words, named_entities, lenght_min))
        return texts_tokens
