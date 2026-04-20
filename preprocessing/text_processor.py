import spacy
# Load spaCy globally and avoids reloading of models which improves performance 
nlp = spacy.load("en_core_web_md", disable=["ner"])
def process_text(text):
    # Process raw text into spaCy document object
    # Doc=is a structured representation
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
  