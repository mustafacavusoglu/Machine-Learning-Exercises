# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:31:54 2020

@author: mustdo
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math as mt
import random


veriler = pd.read_csv("Ads_CTR_Optimisation.csv")





N = 10000
d = 10

toplam = 0
secilenler = []
birler = [0] * 10
sıfırlar = [0] *10
for  n in range(N):
    ad = 0
    max_th = 0
    
    for  i in range(d):
        rasbeta = random.betavariate(birler[i] +1 ,sıfırlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    if odul == 1:
        birler[ad] = birler[ad] + 1
    else:
        sıfırlar[ad] = sıfırlar[ad] + 1
        
    
    toplam = toplam + odul


plt.hist(secilenler)
plt.show()
print(toplam)



