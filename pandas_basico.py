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

print df.T #transposição do DF

print df.describe()

print df.sort_index(axis = 0, ascending=False) #ordena as linhas em ordem descrescente
print df.sort_index(axis = 1, ascending=False) #ordena as colunas em ordem descrescente
print df.sort_values(by=['B'], ascending=False) #ordena pela coluna B em ordem descrescente

print df['A'] #somente os valores da coluna A
print df[['A','B']] #somente os valores das colunas A e B
print df[0:3] #pega da linha 0 a 3 exclusive

#Utiliza os valores do Index e das colunas. Por isso não é exclusive
print df.loc[dates[0]]
print df.loc[:,['A','B']]
print df.loc['20130101':'20130103', ['A','B']]
print df.loc['20130101', ['A','B']]

#Utiliza o index (zero-based), por isso é xclusivo
print df.iloc[3]
print df.iloc[3:5]

#Filtra o DF onde todo o valor da coluna A for maior que zero
print df[df.A > 0]
print df[df.A > 0]['A']
print df[df > 0]

df2 = df.copy()
print df2
df2['E'] = ['one', 'two', 'three', 'four', 'five', 'six']

print df2[df2['E'].isin(['two', 'three'])]

print df[(df['A'] > 0) & df['D'] > 0][['B','C']]
print df[(df['A'] > 0) | df['D'] > 0]

#Estatística
print df.mean() #média de todas as colunas
print df.mean(0) #média de todas as colunas
print df.mean(1) #média de todas as linhas

df.apply(np.cumsum) #soma os valores de forma cumulativa

s = pd.Series(np.random.randint(0,7, size=10))
print s
s.value_counts

print df.groupby('A').sum()

s.plot()

df.to_csv('x.csv')
x = pd.read_csv('x.csv')