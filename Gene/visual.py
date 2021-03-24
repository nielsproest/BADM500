import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

TBL = {
	"LumA": "green",
	"LumB": "blue",
	"Normal": "magenta",
	"Her2": "red",
	"Basal": "black",
}

dt = pd.read_csv("../HiSeq/master.csv",sep=";")
#print(dt.head())

cnt = {
	"LumA": [],
	"LumB": [],
	"Normal": [],
	"Her2": [],
	"Basal": [],
}

#gene = "DLGAP5"
while(True):
	plt.clf()
	gene = input("Give gene: ")

	for i, v in enumerate(dt[gene]):
		label = dt.loc[i]["PAM50_Label"]
		cnt[label].append(v)

	for i,k in enumerate(cnt):
		first = False
		for v in cnt[k]:
			plt.scatter(i, v, label=k if not first else None, color=TBL[k], 
						marker= "*", s=30) 
			first = True

	# x-axis label 
	plt.xlabel('x - axis') 
	# frequency label 
	plt.ylabel('y - axis') 
	# plot title 
	plt.title("{} gene plot!".format(gene)) 
	# showing legend 
	plt.legend() 

	# function to show the plot 
	#plt.show()
	plt.savefig("gene_{}.png".format(gene))
