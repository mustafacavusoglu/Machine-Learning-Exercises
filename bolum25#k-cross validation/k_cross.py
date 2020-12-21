# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:41:07 2020

@author: mustdo
"""
import numpy as np 
import pandas as pd 


data = pd.read_csv("Social_Network_Ads.csv")

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3,random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

from sklearn.svm import SVC
svc = SVC(kernel = "rbf", random_state=0)
svc.fit(X_train,y_train)
y_pred2 = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)#dtc değerler
cm2 = confusion_matrix(y_test,y_pred2)#svc değerler
print(cm)
print(cm2)

from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = svc, X=X_train, y = y_train, cv = 4)
cvs2 = cross_val_score(estimator = dtc, X=X_train, y = y_train, cv = 4)

print("SVC başarısı")
print(cvs.mean())
print(cvs.std())
print("DTC başarısı")
print(cvs2.mean())
print(cvs2.std())