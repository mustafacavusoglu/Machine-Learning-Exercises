#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("sepet.xls", header = None)

t = []

for i in range(7501):
    t.append([str(veriler.values[i,j]) for j in range(20)])



from apyori import apriori
kurallar = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)

print(list(kurallar))
