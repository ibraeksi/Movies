import numpy as np
import pandas as pd

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_sentences(sentence):
    """
    Performs cleaning to given sentences and returns lemmatized sentence for processing
    sentence = User input for movie plot
    """
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    # Lemmatizing the verbs
    verb_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v") # v --> verbs
        for word in tokenized_sentence if not word in stop_words
    ]

    # 2 - Lemmatizing the nouns
    noun_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "n") # n --> nouns
        for word in verb_lemmatized if not word in stop_words
    ]

    cleaned_sentence = ' '.join(word for word in noun_lemmatized)

    cleaned_sentence = cleaned_sentence.strip()

    return cleaned_sentence
