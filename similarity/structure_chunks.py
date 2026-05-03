def structure_chunks(chunks):
    structured=[]
    pointer=0
    
    for i,chunk in enumerate(chunks):
        size=len(chunk)
        start=pointer
        end=pointer+size-1
        if isinstance(chunk, list):
            text = " ".join(chunk)
        else:
            text = chunk
        chunk_id=i
        sentences=chunk
        text=text
        start_index=start
        end_index=end
        
        structured.append({"chunk_id":chunk_id,"sentences":sentences,"text":text,"start_index":start_index,"end_index":end_index})
        pointer+=size
    return structured