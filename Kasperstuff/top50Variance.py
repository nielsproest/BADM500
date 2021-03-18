# evaluate pca with logistic regression algorithm for classification
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pandas as pd
# define dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')
#print(df.head())

#feature_cols = [df.columns.values[x] for x in range(1,len(df)-1)]
#X = df[feature_cols] # Features
#y = df[df.columns.values[-1]] # Target variable


# var() is variance, sorting by highest variance first.
variance = df.var().sort_values(ascending=False)
# taking the indexes from top50 and reinserting ID's into the index
variance50 = variance[:50].index.insert(0, "sampleID")
# Adding back in the labels (optional)
variance50 = variance50.insert(len(variance50), "PAM50_Label")
# filtering the data by the genes in the top 50
# Result: 1144 rows x 50 columns
data50 = df.filter(variance50)

df = pd.DataFrame(np.array(data50), columns=variance50)
df.to_csv("masterTop50Variance.csv", index=False, sep=";")