# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:25:28 2018

@author: alu2015111446

K-means algorithm
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

data = load_iris()
df = pd.DataFrame(data['data'],
                  columns=data['feature_names'])

from sklearn.cluster import KMeans
cl = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300)
cl = cl.fit(df)

labels = cl.labels_
centroids = cl.cluster_centers_

print centroids
print labels


#Para testar medidas de similaridade
from sklearn.metrics import silhouette_score, calinski_harabaz_score

for k in xrange(2,11):
    cl = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    cl = cl.fit(df)
    labels = cl.labels_
    score_k_means = silhouette_score(df, labels)
    score_km_ca = calinski_harabaz_score(df, labels)
    print k, score_k_means, score_km_ca
    
    

#------ CLUSTER HIERÁRQUICO NO SCIPY
    
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt

Z = linkage(df, 'single') #single, complete, ward
print Z

dendrogram(Z)
plt.plot

k=3
clusters = fcluster(Z, k, criterion='maxclust')
print clusters

#Como avaliar os clusters
#========================

#Normalmente existem 3 tipos de medidas

#Externos: avalia o grau entre a estrutura encontrada e a estrutura conhecida (se já existem as classes)
#          Rand Index, Jaccard, Fowlkes-Mallows
#Internos: avalia o grau entre a estrutura encontrada e a estrutura dos dados
#          SSE (soma dos erros ao quadrado)
#Relativos: avalia entre duas ou mais estruturas qual a melhor
#          Silhueta, Davis-Bouldin, ...

#Silhueta - varia entre -1 e 1
#       -1 = cluster ruim
#       0 e 0.5 = clusters não definidos
#       1 = cluster bem definidos e compactos

# --- Comparando duas partições com a silhueta ---
from sklearn.metrics import silhouette_score
score_k_means = silhouette_score(df, labels)
print score_k_means

score_hierarquico = silhouette_score(df, clusters)
print score_hierarquico