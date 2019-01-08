# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:36:40 2019

@author: Sankar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:16:15 2018

@author: Sankar
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m

#import the dataset
dataset=pd.read_csv('data.csv')
X=dataset.iloc[:,2:-1].values
Y=dataset.iloc[:,1:2].values
#Y=Y.reshape(569,)
'''#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_1=LabelEncoder()
X[:,1]=labelencoder_1.fit_transform(X[:,1])
labelencoder_2=LabelEncoder()
X[:,2]=labelencoder_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]'''

#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
Y[:,0]=labelencoder.fit_transform(Y[:,0])
Y=Y.astype('int')
Y=Y.reshape(569,)

#splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/5,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier=Sequential()

#input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=30))

#hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid',input_dim=30))

#comiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
classifier.fit(X_train,Y_train,batch_size=30,nb_epoch=100)

#predicting the test set results
Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

s=int(m.sqrt(cm.size))
sum1=0
sum2=0 

for i in range(0,s):
    for j in range(0,s):
            if i==j:
                sum1 = sum1 + cm[i][j]
            else:
                sum2 = sum2 + cm[i][j]
                
total=sum1+sum2                
Accuracy=(sum1/total)*100            
print("The accuracy for the given test set is " + str(float(Accuracy)) + "%")
