import pymupdf
import re
import nltk
import os

# Guard NLTK downloads to only run if data is missing
_NLTK_DATA = ["wordnet", "punkt", "punkt_tab"]

def _ensure_nltk_data():
    """Download NLTK data if not already present"""
    for pkg in _NLTK_DATA:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

# Call this once at module load
_ensure_nltk_data()

def extract_text(pdf_path: str) -> str:
    """Extract and clean text from PDF"""
    doc = pymupdf.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += "\n\n" + page.get_text("text")

    doc.close()
    
    # Clean and return
    return clean_text(full_text.strip())

def clean_text(text: str) -> str:
    """Normalize whitespace in text"""
    lines = text.split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines]

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)
    print(extract_text(sys.argv[1]))