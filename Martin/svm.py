from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


X = pd.read_csv("mDF05.csv", sep=',')

df = pd.read_csv("matricen05.csv", sep=',')
last = df.iloc[:,-1]
y = np.ravel(last.to_numpy())


print("data shape: ", X.shape)
print("data head: ", X.head)
print("y: ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy_score: ", accuracy_score(y_test,svclassifier.predict(X_test)))
