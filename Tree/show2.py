import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import pstdev, pvariance, stdev, variance

seq_info = pd.read_csv("../BRCA/BRCA_clinicalMatrix_filtered.csv",sep=";")
seq_labels = pd.read_csv("../PAM50/PAM50_filtered.csv",sep=";")
seq_values = pd.read_csv("../HiSeq/HiSeqV2",sep="\t")
#TODO: Switch to master.csv only

grid = list(seq_values["sample"])
#grid=["a","b","c","d","e","f","g","h","i","j"]

graphs = []
label = []


genes = {}
#for i in grid:
#	genes[i] = []

who = {}
for index, row in seq_info.iterrows():
	data_id = row["sampleID"]
	data_res = row["PAM50Call_RNAseq"]
	who[data_id] = data_res

average = {
	"LumA": [],
	"LumB": [],
	"Normal": [],
	"Her2": [],
	"Basal": []
}

ids = list(seq_values.columns.tolist())
ids.pop(0)
for idx, gene_key in enumerate(grid):
	vals = list(seq_values.iloc[idx])
	vals.pop(0) #Remove name
	#print(vals)

	average = {
		"LumA": [],
		"LumB": [],
		"Normal": [],
		"Her2": [],
		"Basal": []
	}

	for k, v in enumerate(vals):
		name = ids[k]
		if name in who:
			pam50_label = who[name]
			if (str(pam50_label) == "nan"):
				continue
			average[pam50_label].append(v)

	genes[gene_key] = [sum(average[key])/len(average[key]) for key in average]
	print(idx,gene_key)
	#if (idx > 20):
	#	break

def dist(L):
	return (max(L) - min(L))/(len(L)-1)

def dist2(L):
	return pstdev(L)

what = []
for i in genes:
	#print(i)
	what.append([dist2(genes[i]),i])

genes_to_keep = 15
what.sort(key=lambda l: l[0])
what = [i[1] for i in what]
what = what[-genes_to_keep:]
input(what)

# plotting a line plot after changing it's width and height 
f = plt.figure() 
f.set_figwidth(2^16) 
f.set_figheight(2^9) 

#x = list(genes.keys())
x = what
#print(x)
for k, v in enumerate(average.keys()):
	y = [genes[i][k] for i in x]
	#print(y)
	plt.plot(x,y,label=list(average.keys())[k])

#for gg,graph in enumerate(graphs):
#	plt.plot(grid,graph,label=list(average.keys())[gg])

plt.legend(loc=3,bbox_to_anchor=(1,0))
#plt.show()

plt.savefig("showre3_15.png")

