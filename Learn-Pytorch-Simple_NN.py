import os
import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

class models (nn.Module):
    def __init__(self, in_features = 4 ,h1= 8, h2=9, out_features = 3 ): #  4 
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
    def forward (self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
dataset = pd.read_csv('IRIS.csv')


device = "cuda" if torch.cuda.is_available() else "cpu" # Check device if this device have a gpu using gpu or if this device hasn't gpu using cpu
le = LabelEncoder() #transform label type String to integer 
X = dataset.drop('species',axis = 1 ) # drop a label data 
y = dataset.species # only take a label kolom 

X = X.values  # Transform Dataframe data to numpy data 
y = le.fit_transform(y) # Change STR label to Int

X = torch.tensor(X,dtype=torch.float32).to(device) # Change X variabel data to continous data 
y = torch.tensor(y,dtype=torch.long).to(device) # Change y variabel data to  int data

X_train , X_test , y_train , y_test = train_test_split(X,y, random_state= 42, test_size=0.2)  # Split  data 

model = models().to(device=device) #  Move data on cpu to Gpu 
loss_fn  = nn.CrossEntropyLoss() # Measure if data has have eror training  Model can be learn back to resolve error training  
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # Measure error training 


ephocs = 100  # Train how many model can be learn dataset 
predict_erors = [] # If model have a error while practicing reading data store at predict_errors variabel list 

for i in range(ephocs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predict_erors.append(loss.item()) # Store and change data toch in gpu to cpu only 
    
    if i % 10 == 0 : 
        print(f"Epoch {i} | Loss: {loss.item():.4f}")


plt.plot(range(ephocs),predict_erors) 
plt.ylabel("loss/errors")
plt.xlabel("ephocs")
plt.savefig('loss_plot.png')

with torch.no_grad():  # Running model without learn 
    y_eval = model.forward(X_test) # Evaluasi result for X_test
    gagal = loss_fn(y_eval , y_test)  # Calculating how many errors are made after the model has learned

benar = 0 
with torch.no_grad():
    for i , data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1},) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}') #  Display the model's prediction results vs the actual answers in one neat row


        if y_val.argmax().item() == y_test[i]: 
            benar +=1
print(f"yang benar ada {benar} yang benar ")

if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model, 'models/iris_model.pkl')
