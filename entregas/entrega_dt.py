"""
Created on Mon Jul 30 14:36:49 2018
@author: Gustavo Emmel e Everton Thomas

"""


import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

random.seed(0)
np.random.seed(0)  # garante que o conjunto de dados seja sempre particionado da mesma forma

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
           'salary']

df = pd.read_csv(
    filepath_or_buffer='adult.csv',
    dtype={
        'age': np.float32,
        'workclass': 'category',
        'fnlwgt': np.float32,
        'education': 'category',
        'education-num': np.float32,
        'marital-status': 'category',
        'occupation': 'category',
        'relationship': 'category',
        'race': 'category',
        'sex': 'category',
        'capital-gain': np.float32,
        'capital-loss': np.float32,
        'hours-per-week': np.float32,
        'native-country': 'category',
        'salary': 'category',
    },
    na_values='?',
    skipinitialspace=True,
    names=columns

)

# display(df)

lb_make = LabelEncoder()
df["native-country"] = lb_make.fit_transform(df["native-country"])
df["sex"] = lb_make.fit_transform(df["sex"])
df["race"] = lb_make.fit_transform(df["race"])
df["marital-status"] = lb_make.fit_transform(df["marital-status"])
df["education"] = lb_make.fit_transform(df["education"])
df["occupation"] = lb_make.fit_transform(df["occupation"])
df["relationship"] = lb_make.fit_transform(df["relationship"])
df["workclass"] = lb_make.fit_transform(df["workclass"])

fact, class_labels = pd.factorize(df['salary'].astype(np.object))

df['salary'] = fact
columns = df.columns.tolist()
columns.pop(columns.index('salary'))
columns.append('salary')
df = df.reindex(columns=columns)
display(df)

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = df[df.columns[:-1]], df[df.columns[-1]]

# utiliza 25% do dataset para teste
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)


dt = tree.DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, Y_train)

predictions = dt.predict(X_test)

print('accuracy score:', accuracy_score(Y_test, predictions))

# accuracy score: 0.8339270359906645

import graphviz
import pydotplus
from IPython.display import Image

dot_data = tree.export_graphviz(
    dt, out_file=None,
    feature_names=df.columns[:-1],  # ignora classe
    class_names=class_labels,
    filled=True, rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
Image(graph.create_png())


