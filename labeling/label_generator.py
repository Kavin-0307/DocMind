import re
import numpy as np
# Basically what labeling functions do is that multiple of them work together
# to give a probability score of importance to each sentence or data point.

def lf_has_number(sentence):
    if(re.search(r'\d+',sentence)):
        return 1
    return 0
def lf_has_keyword(sentence):
    keywords = ['therefore', 'conclude', 'result', 'significant',
                'found', 'show', 'demonstrate', 'key', 'important']
    sentence=sentence.lower()
    if any(word in sentence for word in keywords):
        return 1
    else :
        return 0
def lf_has_length(sentence):
    if(len(sentence.split()))>5:
        return 1
    else:
        return 0

def generate_labels(sentences):
    scores=[]
    labels=[lf_has_number,lf_has_length,lf_has_keyword]
    n=len(sentences)
    L=np.full((n,len(labels)),-1,dtype=int)
    for j,lf in enumerate (labels):
        for i ,sent in enumerate(sentences):
            L[i,j]=lf(sent)
    scores=np.ma.masked_equal(L,-1).mean(axis=1)
    return{
        "label_matrix":L,
        "scores":scores
    }

if __name__ == "__main__":
    sentences = [
        "This system improves accuracy by 20%",
        "error prone",
        "The proposed model shows significant improvement"
    ]

    print(generate_labels(sentences))