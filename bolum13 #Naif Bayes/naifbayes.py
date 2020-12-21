#%%verilerin okunması bölünmesi
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
veriler = pd.read_csv("veriler.csv")
x = veriler.iloc[:,1:4]
y = veriler.iloc[:,4:]
#%%
print(veriler.corr())
#%%test train olarak bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 0)
#%%ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#%%logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski")
knn.fit(X_train,y_train)
y_pred2 = knn.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)
#%%
from sklearn.svm import SVC
svc = SVC(kernel= 'rbf')
svc.fit(X_train,y_train)
y_pred3 = svc.predict(X_test)
cm3 = confusion_matrix(y_test,y_pred3)
print(cm3)
#%%
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred4 = gnb.predict(X_test)
cm4 = confusion_matrix(y_test,y_pred4)
print(cm4) 




