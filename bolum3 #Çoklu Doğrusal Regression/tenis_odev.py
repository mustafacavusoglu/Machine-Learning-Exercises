import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import*
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ohe = OneHotEncoder()
le = LabelEncoder()

veriler = pd.read_csv("odev_tenis.xls")

play = veriler.iloc[:,-1:]



veriler2 = veriler.apply(le.fit_transform) #toplu şekilde label encoder yapar

outlook1 = ohe.fit_transform(veriler2.iloc[:,0:1]).toarray()

havaDurumu = pd.DataFrame(data = outlook1, index = range(14) ,columns = ["overcast","rainy","sunny"])
sonveriler = pd.concat([havaDurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([sonveriler,veriler2.iloc[:,3:5]], axis = 1)
#%% veriyi bölme
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,[0,1,2,3,5,6]],sonveriler.iloc[:,[4]],test_size = 0.33, random_state = 0)

#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

#%%   Backward Elimination
import statsmodels.regression.linear_model as sm #statsmodel.formula.api ----> statsmodels.regression.linear_model 

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,[0,1,2,3,5,6]], axis = 1) #veri tablosuna 1 sütunu ekler

X_l = sonveriler.iloc[:,[0,1,2,3,6]] #etki eden sütunları bulmak için hepsini aldık 

rOls = sm.OLS(endog = sonveriler.iloc[:,[4]], exog = X_l).fit() #OLS fonksiyonu kulllanarak P-values hesaplar

print(rOls.summary())

#%% Backward elimination sonrası tahmin değerlerleri

x_train = x_train.iloc[:,[0,1,2,3,5]]
x_test = x_test.iloc[:,[0,1,2,3,5]]

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)










