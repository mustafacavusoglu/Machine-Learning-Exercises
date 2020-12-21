#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler  = pd.read_csv("mussteriler.xls")

x = veriler.iloc[:,3:].values
#%% #kmeans
from sklearn.cluster import KMeans 

kmeans = KMeans(n_clusters =2,init='k-means++')
kmeans.fit(x)
#print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i ,init = 'k-means++',random_state = 123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)
    
kmeans = KMeans(n_clusters = 4 ,init = 'k-means++',random_state = 123)
y_tahmin  = kmeans.fit_predict(x)

plt.scatter(x[y_tahmin == 0,0],x[y_tahmin == 0,1],s = 100 , c = "red")
plt.scatter(x[y_tahmin == 1,0],x[y_tahmin == 1,1],s = 100 , c = "blue")
plt.scatter(x[y_tahmin == 2,0],x[y_tahmin == 2,1],s = 100 , c = "black")
plt.scatter(x[y_tahmin == 3,0],x[y_tahmin == 3,1],s = 100 , c = "green")

plt.title("kmeans")
plt.show()

"""
plt.plot(range(1,10),sonuclar)
plt.ylabel("WCC değeri")"""
#%% #hiyeyarşik clustering

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = 'ward')
y_tahmin = ac.fit_predict(x)

plt.scatter(x[y_tahmin == 0,0],x[y_tahmin == 0,1],s = 100 , c = "red")
plt.scatter(x[y_tahmin == 1,0],x[y_tahmin == 1,1],s = 100 , c = "blue")
plt.scatter(x[y_tahmin == 2,0],x[y_tahmin == 2,1],s = 100 , c = "black")
plt.scatter(x[y_tahmin == 3,0],x[y_tahmin == 3,1],s = 100 , c = "green")

plt.title("hc")
plt.show()

#%%
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method ='ward'))
plt.show()
