from Tokenization import *
import math
from sklearn.decomposition import NMF

##Length of summary for DUC data-set documents
summary_length_DUC = 100


## Creates binary term-sentence matrix
def binary_TermSent_Matrix(f_name):
    
    termSentFile = ".\\Pre_Processed\\"+f_name.replace('.txt','')+".csv"

    data = np.genfromtxt(termSentFile, dtype='float64', delimiter=',', names=None)
    A = np.asarray(data,dtype = 'float64')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(A[i][j]!= 0):
                A[i][j]=1.0
    
    return A


##Calculates the number of topics into which the document is decomposed
def no_of_topics_DUC(f_name, A):
    inputFile = ".\\Documents\\"+f_name
    
    [words_count,no_of_sentences,sentences] = tokenize_doc(inputFile)

    avg_sentence_length = math.ceil(words_count/no_of_sentences)

    A_sum = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_sum += A[i][j]
        

    ## Number of latent topics
    k = math.ceil(no_of_sentences * A.shape[0]/A_sum)
        
    if (no_of_sentences < k):
         k = no_of_sentences

    return k

##This function decomposes input matrix A using NMF by
##randomly initializing NMF factor matrices W and H
## k is the number of latent topics into which document
## is decomposed
def NMFDecomposition_Random(A,k):

        model = NMF(n_components = k , init = 'random')

        W = model.fit_transform(A) #term-topic
        H = model.components_       # topic - sentence

        np.set_printoptions(suppress=True)
        
        return (W,H)

##This function calculates the sentences' scores 
##using NMF-TR method proposed in Khurana and Bhatnagar, 2019.
def score_TR(A,W,k):
    W_sum = 0.0

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_sum = W_sum + W[i][j]
    W_sum_rows = np.sum(W, axis =1)
        
    terms_imp = np.zeros(A.shape[0],dtype='float64')
        
    for i in range(W.shape[0]):
        terms_imp[i] = W_sum_rows[i]/W_sum

    sentence_score = np.zeros(A.shape[1],dtype = 'float64')
    
    for i in range(A.shape[1]):
        score = 0.0
        for j in range(A.shape[0]):
            score = score + A[j][i]*terms_imp[j]
        sentence_score[i]= score

    return sentence_score


##This function decomposes input matrix A using NMF by
##using NNDSVD initilaization for NMF factor matrices W and H
## k is the number of latent topics into which document
## is decomposed
def NMFDecomposition_NNDSVD(A,k):

        model = NMF(n_components = k , init = 'nndsvd') ## model with NNDSVD initailization

        W = model.fit_transform(A) #term-topic
        H = model.components_       # topic - sentence

        np.set_printoptions(suppress=True)
        
        return (W,H)
