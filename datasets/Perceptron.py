import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("dataset_1.csv", header=None)
dataset.insert(1,'bias',1)
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, -1].values
t=np.where(y==0, -1, y)

plt.figure(1)   
#plt.subplots()
#wTx=np.dot(w_T,X[0])
#del_E=wTx*t[0]
#w=w-learning_parameter*del_E
E=0
w=np.random.rand(3)
w_T=w.T
learning_parameter=0.01

for i in range(X.shape[0]):
    wTx=np.dot(w_T,X[i])
    if wTx*t[i]<0:
        E+=-1*wTx*t[i]
        del_E=X[i]*t[i]
        w=w-learning_parameter*del_E
    if t[i]==1:
        plt.plot(X[i][1],X[i][2],'bo')
    elif t[i]==-1:
        plt.plot(X[i][1],X[i][2],'ro')
        w_T=w.T

#plt.subplots(212)

x= np.linspace(-2,2,100)        
plt.plot(x,-(w_T[0]/w_T[2])-(w_T[1]/w_T[2])*x, 'k')
plt.figure(1)
    
