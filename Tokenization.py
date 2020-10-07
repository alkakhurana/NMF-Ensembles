## Preprocessing Documents
import os
import csv
import string
import textmining
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import tokenize
from nltk.corpus import stopwords


#Removing Punctuation
def strip_punctuation(s):
        table = str.maketrans({key: None for key in string.punctuation})
        return s.translate(table)


def tokenize_doc(inputFile):
        
        file = open(inputFile, 'r')
        
        text = file.read()      
        text = text.replace('\n',' ')

        #Number of words in the text
        words_count = len(word_tokenize(strip_punctuation(text)))
        
        # split in to sentences and store the sentences in a list
        sentences = tokenize.sent_tokenize(text)

        #Original Sentences
        original_sentences = list(sentences)

        return(words_count, len(original_sentences), original_sentences)
        
