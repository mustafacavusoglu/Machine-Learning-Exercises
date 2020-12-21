#Doğrusal Regression
#%%  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#verileri okumak
veriler = pd.read_csv("satislar.xls")

aylar = veriler[["Aylar"]]
#aylar = veriler.iloc[:,0:1]
satislar = veriler[["Satislar"]]
#satislar = veriler.iloc[:,1:2]


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0 )


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

"""
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test =sc.fit_transform(y_test)

"""
#%% model inşası lineer regresyon


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

#%% 

tahmin = lr.predict(x_test)

#%%

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))




