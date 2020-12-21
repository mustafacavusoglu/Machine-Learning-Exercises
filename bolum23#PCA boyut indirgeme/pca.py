# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:40:08 2020

@author: mustdo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("Wine.xls")

x = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state= 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#pca kodlaması

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

print("pca olmadan çıkan sonuçlar")
cm = confusion_matrix(y_test,y_pred)
print(cm)


print("\npca olunca çıkan sonuçlar")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)


print("\nnormal vs pca")
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)









