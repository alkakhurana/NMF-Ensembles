##Code for NMF best experiment
##Based on the size of the ensemble, NMF base model summaries
##are generated and also consensus summaries are generated
##for six combining methods - average, median, quartile,
##voting, ranking and stacking

from Tokenization import *
from NMF import *
import string
import textmining
import os
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import tokenize
from nltk.corpus import stopwords
import numpy as np 
from sklearn.decomposition import NMF
import math
from numpy import genfromtxt
import pandas as pd

## Size of the ensmeble
ensemble_size = 10

## Funtion to rank sentences based on scores
## Rank 1 is assigned to sentence with highest score
def rank_sentences(scores):
    sentence_scores = np.copy(scores)
    
    y=np.argsort(sentence_scores)

    sentence_ranks = np.zeros(no_of_sentences,dtype=int)
    
    for i in range(no_of_sentences):
        sentence_ranks[y[i]] = no_of_sentences - i

    return sentence_ranks

## This function generates base models by randomly initializing NMF
## factor matrices W, H and calculates NMF-TR score of sentences 
## for each base model. The method also generates matrix to be 
## used as input for Stacking ensemble.    
def generateBaseModels(A,k,f):
    
    scores_TR = np.zeros((ensemble_size, A.shape[1]), dtype = 'float64')
    

    W_Stack = np.zeros((ensemble_size*k, A.shape[0]), dtype = 'float64')
    
    index = 0

    for p in range(ensemble_size):
        
        [W, H] = NMFDecomposition_Random(A,k)

            
        np.set_printoptions(suppress=True)
        
        temp_TR = score_TR(A,W,k)

        generate_base_model_summary(temp_TR, f,index)
           
        for i in range(A.shape[1]):
            scores_TR[index][i]=temp_TR[i]

        W_transpose = np.transpose(W) ## topic-term

        for i in range(k):
            for j in range(A.shape[0]):
                W_Stack[p*k+i][j] = W_transpose[i][j]
        
        index+=1   

        np.set_printoptions(suppress=True)

    return(scores_TR, W_Stack)

##This function generates the summaries correspoding
##to a base model
def generate_base_model_summary(temp_TR, f_name,index):
    score_TR = np.copy(temp_TR)
    TR_ranks = np.copy(rank_sentences(score_TR))
    selected_sentences_TR = summary_sentences(TR_ranks)
    summ_name = "system"+str(index+1)+"_"
    write_summary(selected_sentences_TR, f_name, summ_name)

## This function selects the sentences to be included in the summary
## starting with top ranked sentences until summary length is complete.
def summary_sentences(ranks):
    words=0
    selected_sentences = []
    
    for i in range(1,no_of_sentences+1):
            for j in range(no_of_sentences):
                if(ranks[j] == i):
                    if(words < summary_length):
                        words = words + len(word_tokenize(strip_punctuation(sentences[j])))
                        selected_sentences.append(j)
    return selected_sentences

## This function writes the summary to the text file
## Algorithmic summary i.e. the system summaries
## are prefixed with system_ to ease evaluation using
## ROUGE. All the summaries are created in the Summaries
## directory under the main directory "Ensemble Summarization".
def write_summary(selected_sentences,f_name,summ_name):
    
        sorted_sentences = np.sort(selected_sentences)
        
        name = ".\\Summaries\\"+summ_name+f_name
        file_object = open(name,"w")
        
        for i in range(len(selected_sentences)):
                file_object.write(sentences[sorted_sentences[i]].replace('\n',' '))
                file_object.write('\n')
      
        file_object.close()    

## Function to generate summary based on average ensemble    
def averageEnsemble(scores_TR,f_name):

    avg_score_TR = np.zeros(shape=(1,no_of_sentences),dtype = 'float64')
    avg_score_TR = np.average(scores_TR,axis=0)
    TR_ranks = np.copy(rank_sentences(avg_score_TR))
    selected_sentences_TR = summary_sentences(TR_ranks)
    write_summary(selected_sentences_TR, f_name, "system101_")
    

## Function to generate summary based on median ensemble
def medianEnsemble(scores_TR, f_name):
   
    median_score_TR = np.zeros(shape=(1,no_of_sentences),dtype = 'float64')
    median_score_TR = np.percentile(scores_TR,50,axis=0)
    TR_ranks = np.copy(rank_sentences(median_score_TR))
    selected_sentences_TR = summary_sentences(TR_ranks)
    write_summary(selected_sentences_TR, f_name, "system102_")

## Function to generate summary based on quartile ensemble    
def quartileEnsemble(scores_TR, f_name):
    
    quartile_score_TR = np.zeros(shape=(1,no_of_sentences),dtype = 'float64')
    quartile_score_TR = np.percentile(scores_TR,75,axis=0)
    TR_ranks = np.copy(rank_sentences(quartile_score_TR))
    
    selected_sentences_TR = summary_sentences(TR_ranks)
    
    write_summary(selected_sentences_TR, f_name, "system103_")

## Function to generate summary based on voting ensemble
def voting(scores_TR, f_name):
    scores_TR_copy = np.copy(scores_TR)
    TR_ranks = np.zeros((ensemble_size, no_of_sentences), dtype = 'int32')

    for i in range(ensemble_size):
        TR_ranks[i] = rank_sentences(scores_TR_copy[i])
    
    final_TR_ranks = np.zeros(no_of_sentences,dtype=int)
        
    for i in range(no_of_sentences):

        unique_elements_TR,count_elements_TR = np.unique(TR_ranks[:,i],return_counts=True)
        final_TR_ranks[i] = unique_elements_TR[np.argmax(count_elements_TR)]   

    
    selected_sentences_TR = summary_sentences(final_TR_ranks)
   
    write_summary(selected_sentences_TR, f_name, "system104_")
   
## Function to generate summary based on ranking ensemble
def rankingEnsemble(scores_TR, f_name,k):

    scores_TR_copy = np.copy(scores_TR)
    TR_ranks = np.zeros((ensemble_size, no_of_sentences), dtype = 'int32')
    
        
    for i in range(ensemble_size):
        TR_ranks[i] = rank_sentences(scores_TR_copy[i])
    
    TR_count = np.zeros(no_of_sentences, dtype = 'int32')
        
    for i in range(no_of_sentences):
        TR_count[i]=len(list(x for x in TR_ranks[:,i].tolist() if 1 <= x <= k))

    final_TR_ranks = np.zeros(no_of_sentences,dtype=int)
    final_TR_ranks = np.copy(rank_sentences(TR_count))
    
    selected_sentences_TR = summary_sentences(final_TR_ranks)
    
    write_summary(selected_sentences_TR, f_name, "system105_")

## Function to generate summary based on Stacking ensemble
def stackingEnsemble(A, W_Stack, k, f_name):

        W_Stack_copy = np.copy(W_Stack) ## this is topic - term matrix
        
        [Topic_EnsembledTopics, EnsembledTopics_Terms] = NMFDecomposition_NNDSVD(W_Stack_copy,k)

        W_matrix = np.copy(EnsembledTopics_Terms.transpose()) ##this is now term-ensembled_topics

        H_matrix = np.matmul(EnsembledTopics_Terms,A)        
                
        scores_TR = score_TR(A,W_matrix,k)
        
        TR_ranks = np.copy(rank_sentences(scores_TR))

        selected_sentences_TR = summary_sentences(TR_ranks)
        
        write_summary(selected_sentences_TR, f_name, "system106_")
        

## Replace it with the path where the current directory is saved
##os.chdir(".\\NMF-Ensembles")

summary_length = 100.0
           
for f in os.listdir(".\\Documents"):
    print(f)
    inputFile = ".\\Documents\\"+f

    [no_of_words, no_of_sentences, sentences]= tokenize_doc(inputFile)
    
    A = binary_TermSent_Matrix(f)
    k = no_of_topics_DUC(f, A)
    [scores_TR, W_Stack]= generateBaseModels(A,k,f)
    averageEnsemble(scores_TR, f)

    medianEnsemble(scores_TR, f)

    quartileEnsemble(scores_TR,f)

    voting(scores_TR,f)

    rankingEnsemble(scores_TR, f,k)

    stackingEnsemble(A, W_Stack, k, f)



