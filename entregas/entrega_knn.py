# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 04:26:09 2018

@author: Everton Thomas e Gustavo Emmel

#Análise dataset:
    Os atributos são as colunas binarizadas quanto a característica do animal (index):
        - se possui penas, pelos, dentes, etc..
    A classe é a última coluna do dataset, e indica uma categoria a qual o respectivo animal pertence
    
#Análise variação conjunto de teste:
    É possível observar uma relação entre a escolha do tamanho do conjunto de teste e a acurácia obtida.
    Quanto maior o conjunto de teste consequentemente o conjunto de treino será menor. Nesses casos há uma perda
    considerável na acurácia do modelo. Isto exlica-se principalmente devido as poucas instâncias para treinar o algoritmo
    antes de aplicá-lo aos casos de teste.
    *Para melhor avaliar esses cenários retiramos o SHUFFLE do método train_test_split

#Variação de K = 1 a 5
    Verificamos que, no respectivo conjunto de dados, K=1 e K=2 obteram a melhor acurácia (90.47619047619048)
    Conforme K aumenta, identificamos um decréscimo deste valor:
        K value:  1  | Accuracy:  90.47619047619048
        K value:  2  | Accuracy:  90.47619047619048
        K value:  3  | Accuracy:  85.71428571428571
        K value:  4  | Accuracy:  85.71428571428571
        K value:  5  | Accuracy:  76.19047619047619
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def getAccuracy(testset, predictions):
    correct = 0
    for id_test, test in testset.iterrows():
        p = predictions.loc[id_test][0]
        if test[-1] == p:
            correct += 1
    return (correct / float(len(testset))) * 100.0


# prepare data
zoo = pd.read_csv('../datasets/zoo/zoo.csv', nrows=None, header=0)

zoo = zoo.drop(columns=['animal_name'])

trainset, testset, = train_test_split(zoo, test_size=0.2, shuffle=False)

# print trainset
# print testset


#k = 1

for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainset.iloc[:, 0:-1], trainset.iloc[:, -1])
    predicted_knn = knn.predict(testset.iloc[:, 0:-1])
    df_predict = pd.DataFrame(predicted_knn, index=testset.index.values)
    accuracy = getAccuracy(testset, df_predict)
    
    print("K value: ", k, " | Accuracy: ", accuracy)

