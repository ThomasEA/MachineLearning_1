# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:35:47 2018

@author: alu2015111446
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

s = pd.Series([1,3,5,np.nan,6,8], dtype = np.float32)
print s

dates = pd.date_range('20130101', periods=6)
print dates

#Não informando o Index o Pandas vai criar a coluna com 1, 2, 3, 4, ...
df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
print df

#Agora com o Index sendo das datas
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print df

print df.head() #primeiros n-1
print df.tail(3) #cauda (os últimos)
print df.index
print df.columns
print df.values #O Numpy array interno do DataFrame (sem o index e os nomes de colunas)