import pdb
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import nhanes as nhanes

DATA_PATH = "/Users/Artem/Documents/CS 205/NHANES/"

ds = nhanes.Dataset(DATA_PATH)
df = ds.load_all()
print(df.head)

