# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:56:37 2018

@author: Everton Thomas e Gustavo Emmel

O melhor score obtido foi com a seguinte configuração
    Quantidade de iterações: 10
    Alpha: 0.1
    Score: 460.39
    Custo mínimo: 183
    
É possível verificar a ocorrência de underfiting, já que as previsão 
no conjunto de treino não é muito precisa, contemplando inclusive valores negativos para
as casas.

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

def gradient_descent(X, y, thetas, alpha, max_iter):
    costs = np.zeros(max_iter, dtype=np.float64)
    N = len(X)
    J = len(thetas)
    soma = 0.
    thetas_tmp = thetas
    
    prev_cost = cost_function(X, y, hypothesis, thetas)
    
    thetas_final = []
    
    for i in range(max_iter):
        for j in range(J):
            for n in range(N):
                soma += (hypothesis(X[n], thetas) - y[n]) * X[n][j]
    
            thetas_tmp[j] = thetas_tmp[j] - (alpha/len(X)) * soma
        
        thetas = thetas_tmp

        cost = cost_function(X, y, hypothesis, thetas)
        
        costs[i] = cost
        
        if cost < prev_cost:
            thetas_final = thetas
            prev_cost = cost
        

    return thetas_final, costs, np.min(costs)

columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv('datasets/casas/house.data.csv', sep='\t', names=columns)

#1. Seta os valores NAN para a média da coluna
df = nan_to_average(df)

#2. Normaliza os dados, exceto a classe
df = normalize(df, columns)

#3. adiciona x0 ao dataset para viabilizar o algoritmo
df = add_x0(df)

#4. Terminado o préprocessamento, divide DS em treino e teste
trainset, testset = train_test_split(df, test_size=0.3)

#5. Avaliamos algumas features e sua relação com a classe
scatter_features(df['CRIM'], df['MEDV'], 'Taxa crimes per capta', 'Valor médio casas', 'red')
scatter_features(df['AGE'], df['MEDV'], 'Ocupação casas constr. antes 1940', 'Valor médio casas', 'red')
scatter_features(df['RAD'], df['MEDV'], 'Indice acessib. a rodovias', 'Valor médio casas', 'red')
scatter_features(df['DIS'], df['MEDV'], 'Dist. 5 centros empregatícios em Boston', 'Valor médio casas', 'red')
scatter_features(df['RM'], df['MEDV'], 'Número médio de quartos', 'Valor médio casas', 'red')

#6. Aplicamos o Gradiente Descendente no conjunto de treino
X = np.array(trainset.iloc[:,:-1], dtype=np.float32)
y = np.array(trainset.iloc[:,-1], dtype=np.float32)

max_iter = 6
alpha = 0.1
thetas = np.ones(X.shape[1])

cost = cost_function(X, y, hypothesis, thetas)

theta_final, costs, min_cost = gradient_descent(X, y, thetas, alpha, max_iter)

#7. Plot dos custos e do mínimo global (para a quantidade de iterações)
m = np.vstack((np.arange(max_iter), costs))

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(m[0], m[1], c='r')
point, = ax.plot(m[0][np.argmin(m[1])], m[1][np.argmin(m[1])], 'bo')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')

ax.annotate('mínimo global (para as iterações)', 
            xy=(m[0][np.argmin(m[1])], m[1][np.argmin(m[1])]), 
            xytext=(m[0][np.argmin(m[1])] + 2, m[1][np.argmin(m[1])]),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.annotate('Valor custo mínimo: {0}'.format(round(m[1][np.argmin(m[1])]),3),xy=(max_iter/2,m[1][np.argmax(m[1])]))

#8. Faz as predições sobre o dataset de teste
predicted = []
X_test = np.array(testset.iloc[:,:-1], dtype=np.float32)
y_test = np.array(testset.iloc[:,-1], dtype=np.float32)

for p in range(len(X_test)):
    predicted.append(hypothesis(X_test[p], theta_final))

#9. Calcula o score
score = mean_squared_error(y_test, predicted)

print('Score: ', score)

