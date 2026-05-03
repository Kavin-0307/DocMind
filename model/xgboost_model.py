from xgboost import XGBClassifier

def train_xgboost(X_train,y_train,X_val,y_val):
    model=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8)
    model.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=10,verbose=False)
    return model
def predict(model,X):
    return model.predict(X)
