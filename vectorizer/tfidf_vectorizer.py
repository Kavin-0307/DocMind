from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(processed):
    vectorizer=TfidfVectorizer(input='content',max_features=5000,lowercase=True)
    tfidf_matrix=vectorizer.fit_transform(processed["sentences"])
    feature_names=vectorizer.get_feature_names_out()
    return {
        "matrix":tfidf_matrix,
        "feature_names":feature_names,
        "vectorizer":vectorizer
    }

if __name__ == "__main__":
    processed = {
        "sentences": [
            "warehouse monitoring system",
            "manual process error prone"
        ]
    }

    print(vectorize_text(processed))