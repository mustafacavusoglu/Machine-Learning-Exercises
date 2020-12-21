import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


maaslar = pd.read_csv("maaslar_yeni.xls")

d_maaslar = maaslar.iloc[:,2:]
x = d_maaslar.iloc[:,0:3]
y = d_maaslar.iloc[:,-1]
X = x.values
Y = y.values
print(maaslar.corr()) #korelasyon katsayılarını veriyor
#%%# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
import statsmodels.regression.linear_model as sm
model1 = sm.OLS(lr.predict(X),X)
print(model1.fit().summary())
#%% polynomal regression
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 4)
xp = pr.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(xp,y)
model3 = sm.OLS(lr2.predict(pr.fit_transform(X)),X)
print(10*"\t"+"POLY OLS")
print(model3.fit().summary())
#%% Support Vector Regression
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(np.array([Y]).reshape(-1,1))
from sklearn.svm import SVR
svrReg = SVR(kernel = 'rbf')
svrReg.fit(x_olcekli,y_olcekli)
model4 = sm.OLS(svrReg.predict(x_olcekli),x_olcekli)
print(9*"\t"+"SVR OLS")
print(model4.fit().summary())
#%% Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtReg = DecisionTreeRegressor()
dtReg.fit(X,Y)
model5 = sm.OLS(dtReg.predict(X),X)
print(9*"\t"+"DECİSİON TREE OLS")
print(model5.fit().summary())
#%% Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfReg = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
rfReg.fit(X,Y)
model6 = sm.OLS(rfReg.predict(X),X)
print(9*"\t"+"RANDOM FOREST OLS")
print(model6.fit().summary())
#%% Karşılaştırma
#LİNEAR
print("\n"+100*"#")
print(9*"\t"+"LİNEAR OLS")
print(model1.fit().summary())
print("\n"+100*"#")
#POLY
print(10*"\t"+"POLY OLS")
print(model3.fit().summary())
print("\n"+100*"#")
#SVR
print(9*"\t"+"SVR OLS")
print(model4.fit().summary())
print("\n"+100*"#")
#DECİSİON TREE
print(9*"\t"+"DECİSİON TREE OLS")
print(model5.fit().summary())
print("\n"+100*"#")
#RANDOM FOREST
print(9*"\t"+"RANDOM FOREST OLS")
print(model6.fit().summary())
print("\n"+100*"#")





