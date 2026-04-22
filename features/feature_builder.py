from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np
# Converts everything to sparse and then combines them all into one

def build_features(tfidf_matrix,label_matrix,scores):
    label_matrix_sparse=csr_matrix(label_matrix)

    scores_sparse = csr_matrix(scores.reshape(-1, 1))   
    combined=hstack([tfidf_matrix,label_matrix_sparse,scores_sparse])
    return combined

if __name__=='__main__':
    tfidf = csr_matrix([[0.1, 0.3], [0.2, 0.0]])
    
    label_matrix = [[1, 0], [0, 1]]
    scores = [0.5, 0.5]
    print(build_features(tfidf, label_matrix, scores)) 
    