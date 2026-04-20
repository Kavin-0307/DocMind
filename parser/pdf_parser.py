import pymupdf
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab') # Required for newer versions of NLTK
def extract_text(pdf_path):
    doc=pymupdf.open(pdf_path)
    full_text=""
    for page in doc:
        full_text=full_text+"\n\n"+page.get_text("text")
    
    part=full_text[:500]
    tokens1=nltk.word_tokenize(part)
    wnl=nltk.WordNetLemmatizer()
    x=[wnl.lemmatize(t) for t in tokens1]
    print(x)    
    full_text=full_text.strip()
    # full_text.lower() could be done if we wanted to tokenize
    doc.close()
    full_text = clean_text(full_text)
    return full_text

def clean_text(text):
    # Normalize spaces within lines
    lines = text.split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines]
    
    text = "\n".join(cleaned_lines)
    
    # Reduce multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()
if __name__=="__main__":
    pdf_path=r"C:\Users\Kavin\DocMind\IIDTL_C7_group5.pdf"
    extract_text(pdf_path)

