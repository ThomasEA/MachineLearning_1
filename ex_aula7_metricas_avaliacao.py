# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:17:04 2018

@author: alu2015111446

Exercício Aula 7
Métrica de avaliação
"""

import pandas as pd
import numpy as np
from collections import Counter #conta e agrupa itens em uma coleção
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv('RH.csv', nrows=None, header=0, index_col=None)

#RETIRADO POIS O KNN NÃO ACEITA VARIÁVEIS TEXTO
#NA VIDA REAL DEVEM SER DISCRETIZADAS
df = df.drop(['sales','salary'], axis=1)

train_set, test_set = train_test_split(df, test_size=0.2)

n_folds=5

X = train_set.drop(['left'], axis=1)
y = train_set['left']

skf = StratifiedKFold(n_splits=n_folds)

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    predictions = []
    k = 3 #NÃO PODE SER PAR NO KNN
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted_knn = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, predicted_knn)
    print accuracy

