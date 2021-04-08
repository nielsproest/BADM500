import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

df = pd.read_csv("matricen50.csv")
print("data shape: ", df.shape)
print("data head: ", df.head)
first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1] # last column of data frame (id)
df = df.iloc[:, :-1]

X = df.values
print("data X: ", X)
y = np.ravel(last.to_numpy())
print("y: ", y)

clf = SVC(kernel='linear', C=1, random_state=42)
cv = ShuffleSplit(n_splits=10, test_size=0.30)
scores = cross_val_score(clf, X, y, cv=cv)

print("")
print("Cross validation scores: ", scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("")
