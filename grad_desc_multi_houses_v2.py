# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:56:37 2018

@author: Everton Thomas e Gustavo Emmel

Dentro do solicitado pelo trabalho, o melhor score obtido foi:
    alpha = 0.1         # taxa de aprendizado
    threshold = 0.001   # diferença aceitável entre custos
    batch_size = 8      # tamanho do batch
    max_epoch = 10      # máximo número de iterações permitido
    
    Score obtido: 50.38
    Valor custo mínimo obtido: 26.0

Outros testes foram realizados, e o melhor score obtido foi com as configurações:
    alpha = 0.1         # taxa de aprendizado
    threshold = 0.001   # diferença aceitável entre custos
    batch_size = 48     # tamanho do batch
    epoch = 0
    max_epoch = 50     # máximo número de iterações permitido
    
    Score obtido: 14.04
    Valor custo mínimo obtido: 11.0
    

É possível verificar a ocorrência de underfiting quando utilizamos valores
baixos para o batch (entre 1 e 5), já que os valores preditos não se assemelham
com os valores reais.

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def nan_to_average(X):
    if X.isnull().values.any():
        cols = X.columns.tolist()
        for x in cols:
            col_mean = np.mean(X[x])
            X[np.isnan(X[x])] = col_mean
    
    return X
    
def normalize(X, columns):
    columns[:-1]

    scaled_features = StandardScaler().fit_transform(X.iloc[:,:-1].values)
    scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=columns[:-1])
    scaled_features_df['MEDV'] = X['MEDV']
    X = scaled_features_df
    return X

def add_x0(X):
    x0 = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    X['x0'] = x0
    #reposiciona a última coluna (x0) para a primeira posição
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]
    return X

def scatter_features(x, fx, lblx, lbly, color):
    plt.scatter(x, fx, c=color)
    plt.xlabel('x -> {0}'.format(lblx))
    plt.ylabel('fx -> {0}'.format(lbly))
    plt.show()

def hypothesis(X, thetas):
    return np.dot(X, thetas.T)

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def cost_function(X, y, h, theta):
    N = len(X)
    
    #Soma vetorizada
    soma = np.sum((h(X, theta) - y) ** 2.)
    
    return (1./(2. * float(N))) * soma

def update_theta(X, fx, h, thetas, alpha, idx_theta):
    N = len(X)
    soma = 0.
    
    for i in range(N):
        if idx_theta == 0:
            soma += (h(X[i], thetas) - fx[i])
        else:
            soma += (h(X[i], thetas) - fx[i]) * X[i]
    
    if idx_theta == 0:
        return thetas[idx_theta] - ((alpha * (1./float(N))) * soma)
    else:
        return (thetas[idx_theta] - ((alpha * (1./float(N))) * soma))[idx_theta]

def gradient_descent(X, y, thetas, alpha, max_epoch, threshold, batch_size):
    costs = []
    epoch = 0
    prev = np.inf  # custo anterior
    curr = cost_function(X, y, hypothesis, thetas)  # custo atual

    J = len(thetas)

    while (abs(curr - prev) > threshold) and (epoch < max_epoch):
        bc = 0  # contador de quantas instâncias passaram pelo batch
        tmp = np.zeros(len(thetas), dtype=np.float64)
                
        for i in range(batch_size):
            X_local = X[bc:(bc + batch_size)]
            fx_local = y[bc:(bc + batch_size)]

            for j in range(J):
                tmp[j] = update_theta(X_local, fx_local, hypothesis, thetas, alpha, j)

            thetas = tmp

            bc += 1

        prev = curr
        curr = cost_function(X, y, hypothesis, thetas)
        costs.append(curr)
        epoch += 1

    return thetas, costs, np.min(costs)

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv('datasets/casas/house.data.csv', sep='\t', names=columns)

#1. Seta os valores NAN para a média da coluna
df = nan_to_average(df)

#2. Normaliza os dados, exceto a classe
df = normalize(df, columns)

#3. adiciona x0 ao dataset para viabilizar o algoritmo
df = add_x0(df)

#4. Terminado o préprocessamento, divide DS em treino e teste
trainset, testset = train_test_split(df, test_size=0.2)

#5. Avaliamos algumas features e sua relação com a classe
scatter_features(df['CRIM'], df['MEDV'], 'Taxa crimes per capta', 'Valor médio casas', 'red')
scatter_features(df['AGE'], df['MEDV'], 'Ocupação casas constr. antes 1940', 'Valor médio casas', 'red')
scatter_features(df['RAD'], df['MEDV'], 'Indice acessib. a rodovias', 'Valor médio casas', 'red')
scatter_features(df['DIS'], df['MEDV'], 'Dist. 5 centros empregatícios em Boston', 'Valor médio casas', 'red')
scatter_features(df['RM'], df['MEDV'], 'Número médio de quartos', 'Valor médio casas', 'red')

#6. Aplicamos o Gradiente Descendente no conjunto de treino
X = np.array(trainset.iloc[:,:-1], dtype=np.float32)
y = np.array(trainset.iloc[:,-1], dtype=np.float32)

alpha = 0.1         # taxa de aprendizado
threshold = 0.001   # diferença aceitável entre custos
batch_size = 8      # tamanho do batch
epoch = 0
max_epoch = 10     # máximo número de iterações permitido
    
thetas = np.ones(X.shape[1])

theta_final, costs, min_cost = gradient_descent(X, y, thetas, alpha, max_epoch, threshold, batch_size)

#7. Plot dos custos e do mínimo global (para a quantidade de iterações)
m = np.vstack((np.arange(len(costs)), costs))

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(m[0], m[1], c='r')
point, = ax.plot(m[0][np.argmin(m[1])], m[1][np.argmin(m[1])], 'bo')
ax.set_xlabel('Épocas')
ax.set_ylabel('Custo')

ax.annotate('mínimo global (para as iterações)', 
            xy=(m[0][np.argmin(m[1])], m[1][np.argmin(m[1])]), 
            xytext=(m[0][np.argmin(m[1])] + 2, m[1][np.argmin(m[1])]),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.annotate('Valor custo mínimo: {0}'.format(round(m[1][np.argmin(m[1])]),3),xy=(len(costs)/2,m[1][np.argmax(m[1])]))

#8. Faz as predições sobre o dataset de teste com a função aprendida
predicted = []
X_test = np.array(testset.iloc[:,:-1], dtype=np.float32)
y_test = np.array(testset.iloc[:,-1], dtype=np.float32)

for p in range(len(X_test)):
    predicted.append(hypothesis(X_test[p], theta_final))

#9. Calcula o score
score = mean_squared_error(y_test, predicted)

print('Score: ', score)


