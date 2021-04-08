from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np




""" df = pd.read_csv("matricen05.csv", sep=',')
last = df.iloc[:,-1]
y = np.ravel(last.to_numpy()) """

df = pd.read_csv("matricen50.csv")
print("data shape: ", df.shape)
print("data head: ", df.head)
first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1] # last column of data frame (id)
#last.to_csv("last.csv")
df = df.iloc[:, :-1]
#df.to_csv('file.csv')


X = df.values
print("data X: ", X)
y = np.ravel(last.to_numpy())

print("y: ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

deFunc = svclassifier

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy_score: ", accuracy_score(y_test,svclassifier.predict(X_test)))


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    print(names[::-1][0:top])
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.savefig("SVM-feature-importance2.png")

# whatever your features are called


# Specify your top n features you want to visualize.
# You can also discard the abs() function 
# if you are interested in negative contribution of features
f_importances(abs(svclassifier.coef_[0]), feature_labels, top=40)



""" for clf in [SVC(kernel='linear'), SVC(kernel='poly'),
            RandomForestClassifier(n_estimators=100), LogisticRegressionCV()]: """
""" for clf in [RandomForestClassifier(n_estimators=100)]:
    print(clf.__class__.__name__, '' if not hasattr(clf, 'kernel') else clf.kernel)
    clf.fit(X_train, y_train)
    clean_dataset_score = clf.score(X_test, y_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    for index in range(X.shape[1]):
        X_train_noisy = X_train.copy()
        np.random.shuffle(X_train_noisy[:, index])
        X_test_noisy = X_test.copy()
        np.random.shuffle(X_test_noisy[:, index])
        clf.fit(X_train_noisy, y_train)
        noisy_score = clf.score(X_test_noisy, y_test)
        print(df.columns.values[index], clean_dataset_score - noisy_score,
              clean_dataset_score, noisy_score)
    print('') """