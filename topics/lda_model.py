from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
def train_lda(documents: list[str], num_topics: int):
    vectorizer=CountVectorizer(max_features=5000)
    X=vectorizer.fit_transform(documents)
    clf=LatentDirichletAllocation(n_components=num_topics,random_state=42)
    topic_distribution=clf.fit_transform(X)
    feature_names=vectorizer.get_feature_names_out()
    return {"lda_model":clf,"vectorizer":vectorizer, "feature_names": feature_names,"topic_distribution":topic_distribution}
def get_topics(lda_model, feature_names, top_n_words: int):
    topic_word_matrix=lda_model.components_
    topics=[]
    for topic in topic_word_matrix:
        top_indices=topic.argsort()[::-1][:top_n_words]
        topic_words=[feature_names[i] for i in top_indices]
        topics.append(topic_words)
    return topics