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

"""data = dt.values #dt.to_numpy() in old versions
X = np.array([i[1:-1] for i in data])
y = np.array([i[-1] for i in data])
feature_cols = list(dt.columns)[1:-1]"""

#The martin way
first_column = dt.columns[0]
X = dt.drop([first_column], axis=1)
feature_labels = X.columns
y = X.iloc[:,-1]
X=X.iloc[:, :-1]
print(X)
print(y)

y_labels = [ 
	'Normal',
	'Basal',
	'Her2',
	'LumA',
	'LumB'
]

y = np.array([y_labels.index(i) for i in y])

if False:
	with open("out.txt", "w") as f:
		def startified_attempt(depth=None):
			print("Classifying...")
			model = XGBClassifier(max_depth=depth, use_label_encoder=False)
			kfold = StratifiedKFold(shuffle=True, random_state=0) #n_splits=10, 
			results = cross_val_score(model, X, y, cv=kfold)
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

if True:
	clf = XGBClassifier(max_depth=5)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)

	#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("Accuracy {}: {:.2f}".format(5, acc))

	#plot_tree(clf, label=y_labels) #, feature_names=feature_cols
	#plt.savefig("ga2_{}_{:.6f}.png".format(5, acc), dpi=600)

	#plot_importance(clf, max_num_features=50)
	#plt.savefig("ga2_{}_importance_table50.png".format(5), dpi=600, height=0.8)

	"""things = []
	for col,score in zip(X_train.columns,clf.feature_importances_):
		things.append([col,score])
	things.sort(key=lambda x: x[1])
	things.reverse()
	print(things[:50])"""
	results=pd.DataFrame()
	results['columns']=X.columns
	results['importances'] = clf.feature_importances_
	results.sort_values(by='importances',ascending=False,inplace=True)

	print(results[:50])




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