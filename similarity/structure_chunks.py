def structure_chunks(chunks):
    structured=[]
    pointer=0
    
    for i,chunk in enumerate(chunks):
        text=" ".join(chunk) if isinstance(chunk,list) else chunk
        structured.append({"chunk_id":i,"text":text,"start_index":i,"end_index":i})
        
    return structured