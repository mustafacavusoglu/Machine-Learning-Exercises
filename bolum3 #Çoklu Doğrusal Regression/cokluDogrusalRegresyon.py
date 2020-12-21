#%%  eksik veri
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import*
from sklearn.impute import SimpleImputer

veriler = pd.read_csv("eksik.csv")
"""
imputer = SimpleImputer(missing_values = "NaN",strategy = "mean")

Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])

Yas[:,1:4] = imputer.transform(Yas[:,1:4])

print(Yas)
"""
# %%  veri kategorileştirme

cinsiyet = veriler.iloc[:,-1:].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ohe = OneHotEncoder()
le = LabelEncoder()

#cinsiyet = le.fit_transform(cinsiyet)
cinsiyet = ohe.fit_transform(cinsiyet).toarray()


#%% veri hazırlama düzenleme birleştirme
    
ulke = veriler.iloc[:,0:1]
ulke = ohe.fit_transform(ulke).toarray()

boy = veriler.iloc[:,1].values

Yas = veriler.iloc[:,0:4]
sonuc = pd.DataFrame(data = ulke, index = range(22), columns =["fr","tr","us"])

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ["boy","kilo","yas"])

sonuc3 = pd.DataFrame(data = cinsiyet[:,:1], index = range(22), columns = ["cinsiyet"])

s = pd.concat([sonuc,sonuc2], axis = 1)

s2 = pd.concat([s,sonuc3],axis = 1)

#%% veriyi bölme

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)

#%%

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

boy = s2.iloc[:,3:4].values


sol = s2.iloc[:,:3] #ülkeler
sag = s2.iloc[:,4:] #geri kalanlar

veri = pd.concat([sol,sag],axis = 1)

#veri'den(bağımız değişken) boyu(bağımlı değişken) tahmin etme
x_train, x_test, y_train, y_test = train_test_split(veri,boy,test_size = 0.33, random_state = 0) 

reg2 = LinearRegression()

reg2.fit(x_train,y_train) # x'den y'yi tahmin etmek için kullan

y2_pred = reg2.predict(x_test) #x_test'ten y_testi tahmin etme

#%%
import statsmodels.regression.linear_model as sm #statsmodel.formula.api ----> statsmodels.regression.linear_model 

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1) #veri tablosuna 1 sütunu ekler

X_l = veri.iloc[:,[0,1,2,3,4,5]].values #etki eden sütunları bulmak için hepsini aldık 

rOls = sm.OLS(endog = boy, exog = X_l).fit() #OLS fonksiyonu kulllanarak P-values hesaplar

print(rOls.summary())

#%%

X_l = veri.iloc[:,[0,1,2,3,5]].values #P-values en yüksek olanı çıkardık

rOls = sm.OLS(endog = boy, exog = X_l).fit()

print(rOls.summary())

#%%

X_l = veri.iloc[:,[0,1,2,3]].values #P-valueslerin hepsi 0 oldu <---> backward elimination algoritması

rOls = sm.OLS(endog = boy, exog = X_l).fit() 

print(rOls.summary())










