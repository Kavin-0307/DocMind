from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve,auc
import joblib
import numpy as np
def apply_threshold(probabilities,threshold:float):
    return (np.array(probabilities) >= threshold).astype(int)

def evaluate_model(y_true,y_pred):
    f1=f1_score(y_true,y_pred)
    precision=precision_score(y_true, y_pred,zero_division=np.nan)
    recall=recall_score(y_true,y_pred)
    class_rep=classification_report(y_true,y_pred)
    return {"f1_score":f1,"precision_score":precision,"recall_score":recall,"classification_report":class_rep}


def compute_pr_auc(y_true,probabilities):
    precision,recall,thresholds=precision_recall_curve(y_true,probabilities)
    area_under_curve=auc(recall,precision)
    return {"pr_auc":area_under_curve,"precision":precision,"recall":recall,"thresholds":thresholds}
def save_model(model, filepath: str):
    joblib.dump(model,filepath)
    
def load_model(filepath: str):
    return joblib.load(filepath)