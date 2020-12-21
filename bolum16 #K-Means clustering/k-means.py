#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler  = pd.read_csv("musteriler.xls")

x = veriler.iloc[:,3:]
#%%,
from sklearn.cluster import KMeans 

kmeans = KMeans(n_clusters =2,init='k-means++')

kmeans.fit(x)

#print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,10):
    kmeans = KMeans(n_clusters = i ,init = 'k-means++',random_state = 123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,10),sonuclar)
plt.ylabel("WCC değeri")
plt.show()

