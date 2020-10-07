# NMF-Ensembles
Python (v 3.7.4, 32-bit) implementation of NMF based Ensembles for Extractive Single Document Summarization.  
**Author**:  Alka Khurana  
**Acknowledgement**: Vasudha Bhatnagar 


## Pipeline:  
1. Clone the repository.
2. Put all the documents to be summarized in 'Documents' folder.
3. In all .py file change the current directory path to the path in your system where the repository is cloned.
4. Run 'Preprocessing.py'
5. Run 'Ensembles.py' for generating summaries coressponding to different combining methods and different ensemble size.
6. Run 'EnsemblesDUCNMFBest.py' for generating the base model summaries and ensemble summaries.
7. Run 'EnsemblesDUCRandom_Method1.py' and 'EnsemblesDUCRandom_Method2.py' for generating summaries by varying the number of latent topics into which the document is decomposed.
8. 'NMF.py' and 'Tokenization.py' are helper files.
9. System (algorithmic) summaries are stored in 'Summaries' folder.
10. Evaluate the performance of summaries using ROUGE toolkit.
