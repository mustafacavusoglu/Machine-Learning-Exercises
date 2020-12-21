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

ulke = veriler.iloc[:,0:1].values

print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


"""le = LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke)"""

ohe = OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#%% veri hazırlama düzenleme birleştirme

sonuc = pd.DataFrame(data = ulke, index = range(22), columns =["fr","tr","us"])

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ["boy","kilo","yas"])

cinsiyet = veriler.iloc[:,-1]

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])

s = pd.concat([sonuc,sonuc2], axis = 1)

s2 = pd.concat([s,sonuc3],axis = 1)

#%% veriyi bölme

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)

#%% öznitelik dönüşümü 

"""
z = (x-u)/q  ------------> standartlaştırma

z = (x-min(x))/(max(x)-min(x)) ------------> normalleştirme
"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

















