import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

'''
This is an attempt at using k-fold on our own data, using a "LogisticRegression" as stand-in for our models.
'''

#Load data
print("Load data")
df = pd.read_csv("masterTop50Variance.csv",sep=';') # change this to masterNew.csv for real tests, top50 only used for speedy testing 

# data sample
data = np.array(df)
dataNoNames = data[:, 1:-1] # This removes the first column with the Sample ID's and the last column with the PAM50 labels
labels = data[:, -1]        # only pam50 labels

# prepare cross validation
kfold = KFold(10, shuffle=True, random_state=1)
# enumerate splits (Just to illustrate the size of training and test sets)
'''for train, test in kfold.split(data):
    print('train len: %s, \n test len: %s' % (len(data[train]), len(data[test])))'''

# create model to use as example
model = LogisticRegression(max_iter=100000)
# evaluate model with data and labels
X = dataNoNames
y = labels
scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))