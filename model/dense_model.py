import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DenseClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(DenseClassifier,self).__init__()
        self.layer1=nn.Linear(input_dim,hidden_dim)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        return x

def train_dense_model(X_train,y_train,X_val,y_val,epochs=30,lr=0.01):
    X_train=torch.tensor(X_train,dtype=torch.float32)
    y_train=torch.tensor(y_train,dtype=torch.long)

    X_val=torch.tensor(X_val,dtype=torch.float32)
    y_val=torch.tensor(y_val,dtype=torch.long)

    input_dim=X_train.shape[1]
    output_dim=len(np.unique(y_train.numpy()))

    model=DenseClassifier(input_dim,hidden_dim=128,output_dim=output_dim)
    loss_fn=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs=model(X_train)
        loss=loss_fn(outputs,y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs=model(X_val)
            val_loss=loss_fn(val_outputs,y_val)

        if epoch % 5 ==0:
            print(f"Epcoh{epoch}|Train Loss:{loss.item():.4f}|Val Loss: {val_loss.item():.4f}")
    return model
def predict_dense(model, X):

    X = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)

    return preds.numpy()
