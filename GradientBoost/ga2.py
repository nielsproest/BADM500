from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from six import StringIO
import numpy as np
import pandas as pd
import sys

#Load data
print("Load data")
#dt = pd.read_csv("../HiSeq/master.csv",sep=";")
dt = pd.read_csv("../Kasperstuff/masterNew.csv",sep=";")
print(dt.head())

print(len(dt))
print(len(dt.columns))

feature_cols = [dt.columns.values[x] for x in range(1,len(dt.columns)-1)]
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

if False:
	with open("out.txt", "w") as f:
		def startified_attempt(depth=None):
			print("Classifying...")
			model = XGBClassifier(max_depth=depth, use_label_encoder=False)
			kfold = StratifiedKFold(random_state=0) #n_splits=10, 
			results = cross_val_score(model, X, y, cv=kfold)
			#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
			print("Accuracy {}: {:.2f} ({:.2f})".format(depth, results.mean()*100, results.std()*100))
			f.write("Accuracy {}: {:.2f} ({:.2f})\n".format(depth, results.mean()*100, results.std()*100))

		startified_attempt()
		startified_attempt(depth=3)
		startified_attempt(depth=4)
		startified_attempt(depth=5)
		startified_attempt(depth=6)
		startified_attempt(depth=7)
		startified_attempt(depth=8)
		startified_attempt(depth=9)
		startified_attempt(depth=10)


clf = XGBClassifier(max_depth=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Accuracy {}: {:.2f}".format(8, acc))

#plot_tree(clf, label=y_labels) #, feature_names=feature_cols
#plt.savefig("ga2_{}_{:.6f}.png".format(8, acc), dpi=600)

plot_importance(clf, max_num_features=20)
plt.savefig("ga2_{}_importance_table.png".format(8), dpi=600)

if False:
	# Test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

	def attempt(depth=None):
		# Classifier
		clf = XGBClassifier(max_depth=depth)
		clf.fit(X_train, y_train)

		acc = clf.score(X_test, y_test)
		print(acc)

		#plot_tree(clf, feature_names=feature_cols, label=y_labels)
		#plt.show()
		#plt.savefig("ga2_{}_{:.6f}.png".format(depth, acc), dpi=600)

		#plot_importance(clf, max_num_features=10)
		#plt.savefig("ga2_{}_importance_table.png".format(depth), dpi=600)

	while True:
		attempt()
		attempt(depth=3)
		attempt(depth=4)
		attempt(depth=5)

"""
# Visualization. Install graphviz in your system
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(
	model,
	out_file=dot_data, filled=True, rounded=True,
	special_characters=True,
	proportion=False, impurity=False, # enable them if you want
	feature_names = feature_cols, class_names=y_labels,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("ga2_{:.6f}.png".format(acc))
Image(graph.create_png())
"""