from numpy import where
import pandas as pd
import numpy as np# data visualzation
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv("matricen05.csv", sep=',')

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
last = df.iloc[:,-1] # last column of data frame (id)
print("last shape: ", last.shape)
last.to_csv("last05.csv")
print("Prior cut of last: ", df.shape)
df = df.iloc[:, :-1] #cut off last column
#df.add(last, axis=1)

#df = df.drop([first_column], axis=1)
print("Prior reduction shape: ", df.shape)
df.to_csv("mDF05.csv", index=False)

dfCat = pd.Categorical(pd.factorize(last)[0] + 1)
print("After reduction shape: ", dfCat.shape)
dfClasses = pd.DataFrame(np.array(dfCat))
dfClasses.to_csv("Classes05.csv", index=False)
