import pdb
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import nhanes as nhanes

# DATA_PATH = "/Users/Artem/Documents/CS 205/NHANES/"

# ds = nhanes.Dataset(DATA_PATH)
# df = ds.load_all()
# print(df.head)

l = [1,2]

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})

print(df.head())

colv = df['a'].values

mask = np.isin(colv, l)
    
colv[~mask] = 0

print(colv) 

