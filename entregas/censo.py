# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 01:25:14 2018

@author: evert

Entrega 1
Clustering Censo 2005
"""

import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt

columns = ['age',
           'workclass',
           'fnlwgt',
           'education',
           'education-num',
           'marital-status',
           'occupation',
           'relationship',
           'race',
           'sex',
           'capital-gain',
           'capital-loss',
           'hours-per-week',
           'native-country',
           '>50K, <=50K']

#Carrega o dataframe
data = pd.read_csv("data/censo/adult.data", names=columns)

#limpando dados desconhecidos do dataset
data = data[data['workclass'] != ' ?'] 
data = data[data['education'] != ' ?'] 
data = data[data['marital-status'] != ' ?'] 
data = data[data['occupation'] != ' ?'] 
data = data[data['relationship'] != ' ?'] 
data = data[data['race'] != ' ?'] 
data = data[data['sex'] != ' ?'] 
data = data[data['native-country'] != ' ?'] 

#avaliando feature AGE
plt.boxplot(data['age'])
plt.show()
plt.hist(data['age'])
plt.show()
np.percentile(data['age'], q=range(0,100,10))

#retirada dos outliers da feature AGE
data = data.loc[data['age'] < 75]

#avaliando feature AGE após a retirada dos outliers
plt.boxplot(data['age'])
plt.show()
plt.hist(data['age'])
plt.show()
np.percentile(data['age'], q=range(0,100,10))

#discretização da feature workclass
df_dsc = pd.get_dummies(data['workclass'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature education
df_dsc = pd.get_dummies(data['education'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature marital-status
df_dsc = pd.get_dummies(data['marital-status'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature occupation
df_dsc = pd.get_dummies(data['occupation'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature relationship
df_dsc = pd.get_dummies(data['relationship'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature race
df_dsc = pd.get_dummies(data['race'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature sex
df_dsc = pd.get_dummies(data['sex'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature native-country
df_dsc = pd.get_dummies(data['native-country'])

data = pd.concat([data, df_dsc], axis=1)

#discretização da feature >50K, <=50K
df_dsc = pd.get_dummies(data['>50K, <=50K'])

data = pd.concat([data, df_dsc], axis=1)

#remove as colunas originais que foram discretizadas
data = data.drop(columns = ['workclass', 
                  'education', 
                  'marital-status', 
                  'occupation', 
                  'relationship', 
                  'race', 
                  'sex', 
                  'native-country',
                  '>50K, <=50K'])

#limita o dataset para evitar erro de memória
df = data.sample(n = 500)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score

#K-Means
for k in range(2,20):
    cl = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    cl = cl.fit(df)
    labels = cl.labels_
    score_k_means = silhouette_score(df, labels)
    score_km_ca = calinski_harabaz_score(df, labels)
    print(k, score_k_means, score_km_ca)


#Cluster hierárquico
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt

Z = linkage(df, 'single') #single, complete, ward

dendrogram(Z)

k=20
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)