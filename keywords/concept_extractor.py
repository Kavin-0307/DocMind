from keybert import KeyBERT

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_md", disable=["ner"])
    return _nlp

def extract_concepts(text: str, model, top_n: int = 10) -> list[str]:
    kw_model = KeyBERT(model=model)  # reuses the shared SentenceTransformer
    kb_results = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), top_n=top_n
    )
    keybert_phrases = {kw for kw, _ in kb_results}

    nlp = _get_nlp()
    doc = nlp(text)
    noun_phrases = {
        chunk.text.lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) >= 2
    }

    # Prefer noun phrase when a keybert result is a substring of it
    final = set()
    for kw in keybert_phrases:
        absorbed = any(kw in np and np != kw for np in noun_phrases)
        if absorbed:
            continue
        final.add(kw)
    final.update(noun_phrases)

    return list(final)[:top_n]