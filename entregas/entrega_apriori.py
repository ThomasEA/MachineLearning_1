# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:36:49 2018

@author: Everton Thomas e Gustavo Emmel

Suporte: Fração das transações que contém X e Y

Confiança: Frequência que itens em Y aparecem em transações que contem X

Lift: Probabilidade de ocorrência de X e Y independente de um ou outro
    1 = Indica independêcia
    1 < Indica correlação positiva
    1 > Indica correção negativa

"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

retail = pd.read_csv('../datasets/online_retail/online_retail.csv', sep=';')

retail_fr = retail[retail['Country'] == 'France']

dummies = pd.get_dummies(retail_fr['StockCode'])

combine = pd.concat([retail_fr['InvoiceNo'], dummies], axis=1)

transactions = combine.groupby(['InvoiceNo']).sum().reset_index()

transactions = transactions.drop(['InvoiceNo'], axis=1)

#Alguns itens aparecem com indicador 2, ao invés de 0 e 1.
#Então normalizo tudo para 1 quando 
transactions[~transactions.isin([0,1])] = 1

#encontrando regras com no mínimo 5% de suporte
frequent_items = apriori(transactions, min_support=0.05, use_colnames=True)

#encontrar regras com lift maior que 1
rules = association_rules(frequent_items, metric='lift', min_threshold=1)

#10 primeiras regras com maior suporte
rules.sort_values('support', ascending= False).head(10)

#10 primeiras regras com maior confianca
rules.sort_values('confidence', ascending= False).head(10)

#10 primeiras regras com maior lift
rules.sort_values('lift', ascending= False).head(10)

#selecionando regras com lift maior que 2, confianca maior que 0.6 e suporte maior que 0.1
rules[(rules['lift'] >= 2) & (rules['confidence'] >= 0.6) & (rules['support'] >= 0.1 )]