from sklearn.naive_bayes import GaussianNB
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

# Classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Visualization. Install graphviz in your system
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(
	clf,
	out_file=dot_data, filled=True, rounded=True,
	special_characters=True,
	#feature_names = feature_cols, class_names=y_labels,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("bayes.png")
Image(graph.create_png())
