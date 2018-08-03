# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 01:25:14 2018

@authores: Everton Thomas e Gustavo Emmel

Entrega 1
Clustering Censo 2005

Passo-a-passo do pré-processamento:
    1. retirada e dados inexistentes ou incompletos do dataset (?)
    2. avaliação de features a procura de outliers
    3. remoção de outliers da feature 'age'
    4. binarização das features categóricas
    5. investigação da correlação entre as features 'education' e 'education-enum'
    6. remoção da feature 'education' por existir correlação com 'education-enum',
       visando diminuir a quantidade de features e o consequente estouro de memória
    7. remoção das features originais que foram binarizadas no passo anterior
    8. análise da utilização de k-means e algoritmos hierárquicos
    9. normalização das features através de z-score para tentar melhorar os resultados
    10. aplicação de k-means e algoritmos hierárquicos para validar nova estrutura e testar
        se houve melhora no processo
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

#Verifica a correlação entre as features 'education' e 'education-enum'
data['education'] = data['education'].astype('category')
data['education'] = data['education'].cat.codes

plt.scatter(data['education-num'].values, data['education'].values)
plt.xlabel('education')
plt.ylabel('education-num')
plt.show()

pc = np.corrcoef(data['education'].values, data['education-num'].values)
print(pc)

#No passo anterior, apesar do coef. de correlação = 0.34309, 
#pelo gráfico é possível verificar que há somente 1 valor de 'education' para cada 'education-enum'
#então optamos por excluir a feature categorica 'education'
data = data.drop(['education'], axis=1)

#remove as colunas originais que foram discretizadas
data = data.drop(columns = ['workclass', 
                  'marital-status', 
                  'occupation', 
                  'relationship', 
                  'race', 
                  'sex', 
                  'native-country',
                  '>50K, <=50K'])



#limita o dataset para evitar erro de memória
df = data.sample(n = 5000)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score

max_k_silhoute = 0
max_k_calinski = 0
max_score_silhoute = 0
max_score_calinski = 0

#K-Means
for k in range(2,20):
    cl = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    cl = cl.fit(df)
    labels = cl.labels_
    score_km_silhoute = silhouette_score(df, labels)
    score_km_ca = calinski_harabaz_score(df, labels)
    
    if (score_km_silhoute > max_score_silhoute):
        max_score_silhoute = score_km_silhoute
        max_k_silhoute = k
    
    if (score_km_ca > max_score_calinski):
        max_score_calinski = score_km_ca
        max_k_calinski = k
    
    
print('Max. score silhoute: Clusters {} / Score {}'.format(max_k_silhoute, max_score_silhoute))
print('Max. score calinski: Clusters {} / Score {}'.format(max_k_calinski, max_score_calinski))

#Análise:
#Para o cenário atual com k-means, o método da silhueta sugeriu 2 clusters, enquanto o método calinski sugeriu
#19 clusters. Além disso, o score da silhueta (~0.577...) demonstra que os clusters não estão muito bem definidos
#Acreditamos que 19 clusters, conforme sugerido pelo método calinski é um número excessivo de clusters para 
#segmentar os dados.

#Cluster hierárquico
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt

Z = linkage(df, 'single') #single, complete, ward

dendrogram(Z)

Z = linkage(df, 'ward') #single, complete, ward

dendrogram(Z)


k=20
clusters = fcluster(Z, k, criterion='maxclust')

#Análise:
#Para os algoritmos de clusterização hierárquicos, acreditamos que a utilização de WARD se mostrou mais
#eficiente que a 'Single' ou 'Complete', pois parece ter definido melhor os clusters.
#É possível, através de WARD selecionar entre 2 e 4 clusters bem distintos.

##################################
#Em uma tentativa de melhorar os resultados, tentamos normalizar as features:
#   'age', 
#   'fnlwgt', 
#   'capital-gain'
#   'capital-loss'
#   'hours-per-week'
#através do método z-score

df['age'] = (df.age - df.age.mean())/df.age.std(ddof=0)
df['fnlwgt'] = (df.fnlwgt - df.fnlwgt.mean())/df.fnlwgt.std(ddof=0)
df['capital-gain'] = (df['capital-gain'] - df['capital-gain'].mean())/df['capital-gain'].std(ddof=0)
df['capital-loss'] = (df['capital-loss'] - df['capital-loss'].mean())/df['capital-loss'].std(ddof=0)
df['hours-per-week'] = (df['hours-per-week'] - df['hours-per-week'].mean())/df['hours-per-week'].std(ddof=0)

#E aplicamos novamente os algortimos de clusterização

max_k_silhoute = 0
max_k_calinski = 0
max_score_silhoute = 0
max_score_calinski = 0

#K-Means
for k in range(2,20):
    cl = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    cl = cl.fit(df)
    labels = cl.labels_
    score_km_silhoute = silhouette_score(df, labels)
    score_km_ca = calinski_harabaz_score(df, labels)
    
    if (score_km_silhoute > max_score_silhoute):
        max_score_silhoute = score_km_silhoute
        max_k_silhoute = k
    
    if (score_km_ca > max_score_calinski):
        max_score_calinski = score_km_ca
        max_k_calinski = k
    
print('Resultado K-Means após normalização de variáveis:')    
print('Max. score silhoute: Clusters {} / Score {}'.format(max_k_silhoute, max_score_silhoute))
print('Max. score calinski: Clusters {} / Score {}'.format(max_k_calinski, max_score_calinski))

#Análise:
#Em princípio, notamos que talvez as amplitudes e/ou diferentes dimensões entre as features poderia estar
#enviezando o resultado obtido pela medida de avaliação Calinski. Isto porque após normalizar as features,
#ambas as medidas de avaliação (silhueta e calinski) sugeriram a adoção de 2 clusters. Entretanto, os scores
#alcançados reduziram substancialmente, indicando 2 cluster bem indefinidos.

Z = linkage(df, 'single') #single, complete, ward

dendrogram(Z)

Z = linkage(df, 'ward') #single, complete, ward

dendrogram(Z)

#Análise:
#Já para a clusterização hierárquica, a normalização parece ter auxiliado quando utilizado o algoritmo WARD.
#Já para 'Single', o uso da normalização demonstrou clusters muito mais indefinidos do que anteriormente.

#Conclusão:
#No cenário em questão, acreditamos que a melhor opção para clusterizar estes dados seja a adoção do algoritmo
#de clusterização hierárquiva WARD com normalização dos dados.
#Utilzando esse método, e como demonstrado do dendrograma, sugerimos 2 ou 3 clusters.