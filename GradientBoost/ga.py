from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from six import StringIO
import numpy as np
import pandas as pd

#Load data
print("Load data")
dt = pd.read_csv("../HiSeq/master.csv",sep=";")
print(dt.head())

feature_cols = [dt.columns.values[x] for x in range(1,len(dt)-1)]
X = dt[feature_cols] # Features
y = dt[dt.columns.values[-1]] # Target variable

y_labels = [ 
	'Normal',
	'Basal',
	'Her2',
	'LumA',
	'LumB'
]

y = [y_labels.index(i) for i in y]


# Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#TODO: https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/

# Classifier
clf = GradientBoostingClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)

# Get the tree number 42
sub_tree_42 = clf.estimators_[42, 0]
print(len(clf.estimators_))

# Visualization. Install graphviz in your system
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(
	sub_tree_42,
	out_file=dot_data, filled=True, rounded=True,
	special_characters=True,
	proportion=False, impurity=False, # enable them if you want
	feature_names = feature_cols, class_names=y_labels,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("ga_{:.6f}.png".format(acc))
Image(graph.create_png())

