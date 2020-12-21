#%%
import numpy as np
import pandas as pd
import re #regular expression 
import nltk

#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

veriler = pd.read_csv("Restaurant_Reviewss.xls")

derleme = []
for i in range(1000):
    yorum = re.sub("[^a-z A-Z]"," ",veriler["Review"][i])#^işareti olumsuzluk anlamı katar
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derleme.append(yorum)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derleme).toarray()#bağımsız değişken
y = veriler.iloc[:,1].values

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.33, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
gnbY_pred = gnb.predict(x_test)
cm1 = confusion_matrix(y_test,gnbY_pred)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy")
dtc.fit(x_train,y_train)
dtY_pred = dtc.predict(x_test)
cm2 = confusion_matrix(y_test,dtY_pred)
print("gaussian")
print(cm1)
print("decision tree")
print(cm2)