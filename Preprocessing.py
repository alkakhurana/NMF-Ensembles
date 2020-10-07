##Code for pre-processing input documents

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


# Input text tokenization
def tokenization(inputFile):
        
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


## Create term-sentence frequency matrix
def create_terms_and_incidence_matrix(original_sentences):    

        sentences = original_sentences.copy()                
        
        filtered_sentences = []
        # Apply stop word removal to each sentence
        stop_words = set(stopwords.words('english'))
            
        for i in range(len(sentences)):
            temp = []
            word_tokens = word_tokenize(strip_punctuation(sentences[i]))
            for w in word_tokens:
                if w.lower() not in stop_words:
                    temp.append(w.lower())
            filtered_sentences.append(temp)

        tdm = textmining.TermDocumentMatrix()
        for i in range(len(sentences)):
            sent = " ".join(filtered_sentences[i])
            tdm.add_doc(sent)

        temp = list(tdm.rows(cutoff=1))
        vocab = tuple(temp[0])
        
        X = np.array(temp[1:],dtype = 'float64')
        X1 = X.transpose()
        return(X1, vocab)



## write term-sentence matrix to .csv file
def write_in_files(A,original_sentences,vocab,f_name):

        fileObj1 = open(".\\Pre_Processed\\"+f_name.replace('.txt','')+".csv",'w')
        
        np.savetxt(fileObj1, A, fmt='%1.5f', delimiter=",")
        
        fileObj1.close()


## Replace it with the path where the current directory is saved
##os.chdir(".\\NMF-Ensembles")

for f in os.listdir(".\\Documents"):
    print(f)
    inputFile = ".\\Documents\\"+f
    [words_count, no_of_sentences, original_sentences]=tokenization(inputFile)
    [B, vocab] = create_terms_and_incidence_matrix(original_sentences)
    write_in_files(B,original_sentences,vocab,f)

