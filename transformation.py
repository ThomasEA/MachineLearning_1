# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:40:09 2018

@author: alu2015111446

Aula 4
Transformação
"""


#Binarização
from sklearn import preprocessing
import numpy as np
import pandas as pd

#Divide em dois valores, conforme o threshold
X = np.array([[1.,0,-1],[1.,1.,-1],[0.,2.,3]])
print X

binarizer = preprocessing.Binarizer(threshold=0.0).fit(X)
binarizer.transform(X)

#Cria novas variáveis binarizadas
X = np.array([[1.,0,'A'],[1.,1.,'A'],[0.,2.,'B']])
print X

binarizer = preprocessing.MultiLabelBinarizer()
binarizer.fit_transform(X[:,-1])

#Discretização Não supervisionada (não leva em consideração a classe)

#Discretização em intervalos
#Regras de associação, por exemplo, precisam separar valores contínuos em intervalos
tempos = [1.,1.1,1.5,1.7,2.,3.,5.,7.,8.]

#O atributo se transforma em categórico ordinal (com certa ordem)
#Tipos de discretização: largura, frequencia e clusterização
a = np.array([0,1,2,6,6,9,10,10,10,13,18,20,21,21,25])
x = [0,1,2,2,6,6,7,8,9,10,10,10,13,18,20,21,21,25]

t, s = np.percentile(a, [33,66])
print t, s

#Por largura
#Divide o maior valor pelo numero de intervalos. Pode ser afetada por outliers, já que usa o máximo valor
#Por frequencia
#Conta qtd numeros existentes e divide pelo número de intervalos    


#Discretização supervisionada (leva em conta a classe)
#Entropia (objetivo é escolher os intervalos onde será feito o corte, baseado na classe e
#levando em consideração a minimização da entropia)

#scipy.stats.entropy


#Re-escalar: (Normalization, minmax (muito sensivel a outliers), )
a = np.arange(20).reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler as minmax
scaler = minmax()
c = scaler.fit_transform(a)
print c

#Padronização: (standardization, z-score)
#Os dados são padronizados cfe média e desvio padrão, portanto, tb sensíveis a outliers

d = np.arange(20)
e = np.array(map(lambda x: (x - d.mean()) / d.std(), d))
print e

#transformação logarítmica
#x = log(x)