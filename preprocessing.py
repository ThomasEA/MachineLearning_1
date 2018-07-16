# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 00:45:50 2018

@author: evert

Exercícios Pós Data Science
Machine Learning I
Pré-processamento de datasets
"""

import pandas as pd
import numpy as np


### Step 2. Leia o dataset iris
iris = pd.read_csv('datasets/iris/iris.data', sep=',', header=None)

# print iris

### Step 3. Crie as colunas para o dataset
# 1. sepal_length (in cm)
# 2. sepal_width (in cm)
# 3. petal_length (in cm)
# 4. petal_width (in cm)
# 5. class

columns = np.array([['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']])

tmp = np.concatenate((columns, iris), axis=0)

df = pd.DataFrame(data=tmp[1:,:], columns=tmp[0,:])

### Step 4.  Algum dado ausente?
### Não há nenhum dado ausente

### Step 5 Verifique os valores das linhas 10 ate 29 da coluna

u = tmp[9:28,:]
print(u)

k = df.iloc[9:28, [1]]

### Step 6. Certo, modifique os valores NaN por 1.0

df = df.fillna(1.)

### Step 7. Agora delete a coluna da classe

deleted_class = df.iloc[:,:-1]

### Step 8.  Coloque as 3 primeiras linhas como NaN

df.iloc[:3,:]=np.NaN

### Step 9.  Delete as linhas que contenham NaN

df = df.dropna(axis=0)

### Step 10. Reset o index para que ele inicie em 0 novamente

df = df.reset_index(drop=True)

