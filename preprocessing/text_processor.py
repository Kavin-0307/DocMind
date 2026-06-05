import spacy
# Load spaCy globally and avoids reloading of models which improves performance 
_nlp = None
def _get_nlp():
    """Lazy-load spaCy model on first use"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md", disable=["ner"])
    return _nlp

def process_text(text):
    # Process raw text into spaCy document object
    # Doc=is a structured representation
    nlp = _get_nlp()
    doc=nlp(text)
    lemmas=[]#normalized word forms
    sents=[]#sentence level structure
    tokens=[]#original tokens (Preserve actual text)


    #We iterate sentence by sentence to keep structure aligned
    for sent in doc.sents:
        sents.append(sent.text.strip())
        #lemma is the base form running-> run. improves generalization for models
        lemmas.append([token.lemma_ for token in sent if not token.is_punct and not token.is_space])
        # So basically we want to tokenize the entire sentence
        tokens.append([token.text for token in sent if not token.is_punct and not token.is_space])
    
    return {'tokens':tokens,'sentences':sents,'lemmas':lemmas}
if __name__=="__main__":
    sample_text="Warehouse often rely on manual process. error prone"
    print(process_text(sample_text))

import re

def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    merged, buffer = [], ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer); buffer = ""
            continue
        # Heading: ALL CAPS or ends with ":" and short
        is_heading = (stripped == stripped.upper() and len(stripped) > 3) \
                     or (stripped.endswith(":") and len(stripped) < 60)
        if is_heading:
            if buffer: merged.append(buffer); buffer = ""
            merged.append(stripped); continue
        # Merge continuation lines (no sentence-ending punctuation)
        if buffer and not buffer[-1] in ".!?":
            buffer += " " + stripped
        else:
            if buffer: merged.append(buffer)
            buffer = stripped
    if buffer: merged.append(buffer)

    # Noise filter: drop tokens that are pure symbols or <= 2 chars
    _noise = re.compile(r'^\W$|^.{1,2}$')
    cleaned = []
    for line in merged:
        tokens = line.split()
        tokens = [t for t in tokens if not _noise.match(t)]
        if tokens:
            cleaned.append(" ".join(tokens))

    # Normalise whitespace
    text = "\n".join(cleaned)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

  