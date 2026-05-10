def search(query: str, model, index, chunks, k: int):
    vectors = model.encode([query]).astype("float32")

    D, I = index.search(vectors, k)

    results = []

    for idx, score in zip(I[0], D[0]):
        if idx != -1 and idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"] if isinstance(chunks[idx], dict) else chunks[idx],
                "score": 1.0/(1.0+float(score))
            })

    return results