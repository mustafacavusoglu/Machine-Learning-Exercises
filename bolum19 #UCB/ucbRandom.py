#%% random secim

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math as mt


veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random
"""
n = 10000
d = 10
toplam = 0
secilenler = []
for i in range(n):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()
"""
#%% ucb kodlama

N = 10000
d = 10
oduller = [0] * d 
tiklamalar = [0] * d
toplam = 0
secilenler = []
for  n in range(N):
    ad = 0
    max_ucb = 0
    
    for  i in range(d):
        
        if(tiklamalar[i] > 1):
            ortalama = oduller[i]/tiklamalar[i]
            delta = mt.sqrt((3/2)*(mt.log(n)/tiklamalar[i]))
            ucb = delta + ortalama
        else:
            ucb = N * 10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul


plt.hist(secilenler)
plt.show()
print(toplam)

