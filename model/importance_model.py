from sklearn.linear_model import LogisticRegression
import numpy as np
"""
Purpose is to train a logistic regression model to predict sentence importance.
Parameters are the feature matrix X and the scoresfrom the feature_builder
it returns model
"""
def train_model(X,scores):
    threshold=0.7#This is done to basically make sure that only the accepted values are chosen.Logistic Regression is a classifier → needs binary labels
    labels=(np.array(scores)>=0.7).astype(int)#Convert continuous scores (0–1) into binary labels required for classification
    model=LogisticRegression()#used as works well with sparsel TF-IDF
    model.fit(X,labels)#mapping from features → importance pattern
    return model
def predict_importance(model,X):
    probabilities=model.predict_proba(X)#We need the ranking of the probabilities not just the classifications
    return probabilities[:,1]#P(important) is what we actually required
# 