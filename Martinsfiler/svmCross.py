from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np

from statistics import mean, stdev 
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold 
from sklearn import linear_model 
from sklearn import datasets 

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
# Create  classifier object. 
lr = svclassifier.predict(X_test)

   
# Create  classifier object. 




# Create StratifiedKFold object. 
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
   
for train_index, test_index in skf.split(X, y): 
    x_train_fold, x_test_fold = X[train_index], X[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index] 
    lr.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold)) 
   
# Print the output. 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 
print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 





def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.savefig("SVM-feature-importance2.png")
    
f_importances(abs(svclassifier.coef_[0]), feature_labels, top=40)
