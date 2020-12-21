import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


maaslar =pd.read_csv("maaslar.xls")
x = maaslar.iloc[:,1:2]
y = maaslar.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
#print(lr.predict(np.array([6.6]).reshape(1,-1)))
plt.scatter(X,Y,color="red")
plt.plot(X,lr.predict(X))
#%% Ploinomql Regresyon
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=4)
xp = pr.fit_transform(X)
yp = pr.fit_transform(Y)
lr2 = LinearRegression()
lr2.fit(xp,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lr2.predict(pr.fit_transform(X)), color = "black")
plt.show()
#print(lr2.predict(pr.fit_transform(np.array([8]).reshape(1,1)))) #böyle bir dönüşüm yapmak gerekiyor.. 
#%%
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svrReg = SVR(kernel = 'rbf')

svrReg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color = "red")
plt.plot(x_olcekli,svrReg.predict(x_olcekli), color = "black")
plt.show()

print(svrReg.predict(np.array([6.6]).reshape(-1,1)))
#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)

r_dt.fit(X,Y)

plt.scatter(X,Y,color = "red")
plt.plot(X,r_dt.predict(X))
plt.show()

print(r_dt.predict(np.array([6.6]).reshape(-1,1)))
print(r_dt.predict(np.array([11]).reshape(-1,1)))   
#%% Random Forest
k = X -0.4
z = X + 0.5
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_reg.fit(X,Y)
print(rf_reg.predict(np.array([6.6]).reshape(-1,1)))

plt.scatter(X,Y,color = "red")
plt.plot(X,rf_reg.predict(X),color="black")
plt.plot(X,rf_reg.predict(z),color = "green")
plt.plot(X,r_dt.predict(k),color = "yellow")








