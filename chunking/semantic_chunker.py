from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunk_text(sentences:list[str],threshold:float,model):
#Sentence transformers allow the transformation of sentences into vector spaces . They represent sentences as 
#dense vector embeddings.The transformers analyze the words in context in both directions, before and after the current word.
    if not sentences:
        return []
    if(len(sentences))==1:
        return sentences
    
    embeddings=model.encode(sentences)
    current_chunk=[sentences[0]]
    chunks=[]
    for i in range(0,len(sentences)-1):
        similarity=cosine_similarity([embeddings[i]],[embeddings[i+1]])[0][0]#we need to use scalar from matrix
        if(similarity<threshold):
            chunks.append(current_chunk)
            current_chunk=[sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])#We must append the next sentence
    chunks.append(current_chunk)
    return [" ".join(chunk) for chunk in chunks]