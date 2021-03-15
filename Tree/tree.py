# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from concurrent.futures import ThreadPoolExecutor
import threading
import time

#Load data
print("Load data")
dt = pd.read_csv("../HiSeq/master.csv",sep=";")
#print(dt.head())

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

def gen_tree(max_depth, seed=1):
	#Train
	#print("Train")

	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed) # 70% training and 30% test

	# Create Decision Tree classifer object
	#criterion="gini" or "entropy"
	#splitter="best" or "random"
	#max_depth=None or int
	
	clf = DecisionTreeClassifier(
		criterion="entropy",
		max_depth=max_depth,
	)

	# Train Decision Tree Classifer
	clf = clf.fit(X_train,y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test)

	# Model Accuracy, how often is the classifier correct?
	acc = metrics.accuracy_score(y_test, y_pred)
	#print("Accuracy:",acc)

	return clf, acc

def gen_image(clf, max_depth, acc):
	#Create image
	#print("Create image")
	dot_data = StringIO()
	export_graphviz(clf, out_file=dot_data,  
					filled=True, rounded=True,
					special_characters=True,feature_names = feature_cols, class_names=y_labels)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	graph.write_png("tree3_d{}_{:.6f}.png".format(max_depth,acc))
	Image(graph.create_png())

def test(num):
	clf, acc = gen_tree(max_depth=num)
	gen_image(clf, num, acc)
	print(dict(zip(X.columns, clf.feature_importances_)))

def unlimited(num, dbg):
	clf, acc = None, 0

	print("Start")
	try:
		for i in range(50):
			n_clf, n_acc = gen_tree(max_depth=num, seed=None)
			if (n_acc > acc):
				clf, acc = n_clf, n_acc

			if (dbg):
				print("Acc {:.6f}, Iteration {}".format(acc, i), end="\r")
	except KeyboardInterrupt:
		pass

	print("Generating image")
	gen_image(clf, num, acc)

test(3)
test(4)
test(5)
test(None)

"""for i in range(5):
	test(None)
	test(3)
	test(4)
	test(5)"""

"""t1 = threading.Thread(target=unlimited, args=(3,False,))
t2 = threading.Thread(target=unlimited, args=(4,False,))
t3 = threading.Thread(target=unlimited, args=(5,False,))
t4 = threading.Thread(target=unlimited, args=(None,True,))

t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()"""

"""
print("Starting work")
with ThreadPoolExecutor(max_workers=24*2) as e:
	work = []

	for i in range(3000):
		work.append(e.submit(gen_tree, (None,)))
	
	clf_b = None
	acc_b = 0
	for i in work:
		clf, acc = i.result()
		if (acc_b < acc):
			clf_b = clf
			acc_b = acc

	gen_image(clf_b, None, acc_b)
"""
