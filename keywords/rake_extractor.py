from rake_nltk import Rake
"""Purpose:
Extract top keyword phrases from text using RAKE (unsupervised).

Parameters:
- text: raw input text
- top_n: number of top keywords to return

Returns:
- keywords: list of top keyword phrases"""
def extract_keywords(text:str,top_n:int):
    r=Rake()
    r.extract_keywords_from_text(text)
    phrases=r.get_ranked_phrases()
    phrases=phrases[:top_n]#RAKE returns many phrases → you only need most relevant ones
    return phrases
