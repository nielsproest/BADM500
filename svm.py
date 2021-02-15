from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("PCA-95cc.csv", sep=',')
print(data.shape)

first_column = data.columns[0]
X = data.drop([first_column], axis=1)
print(X.head)
y = data.iloc[0:2237,0:1]
#print(y.head)
#print(y.shape)
print(y)
#print(data.columns[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

