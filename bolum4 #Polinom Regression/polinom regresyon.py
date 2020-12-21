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

plt.scatter(X,Y,color="red")
plt.plot(X,lr.predict(X))
#%% polinomal regresyon
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)

xp = pr.fit_transform(X)
yp = pr.fit_transform(Y)

lr2 = LinearRegression()
lr2.fit(xp,Y)
plt.scatter(X,Y,color="red")
plt.plot(X,lr2.predict(pr.fit_transform(X)), color = "black")
plt.show()
