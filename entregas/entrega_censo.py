import pandas as pd
import numpy as np

data = pd.read_csv("data/censo/adult.data")

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

df = pd.DataFrame(data, columns=columns)
df.shape()
x = df.dropna()

