import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

seq_info = pd.read_csv("BRCA_clinicalMatrix",sep='\t')
seq_labels = pd.read_csv("PAM50_filtered.csv",sep=';')
seq_values = pd.read_csv("HiSeqV2",sep='\t')

grid = list(seq_values["sample"])
#grid=["a","b","c","d","e","f","g","h","i","j"]

graphs = []
label = []

average = {
	"LumA": [],
	"LumB": [],
	"Normal": [],
	"Her2": [],
	"Basal": [],
}
for index, row in seq_labels.iterrows():
	data_id = row["Sample ID"]
	data_res = row["PAM50"]
	average[data_res].append(list(seq_values[data_id]))

for key in average:
	tbl = average[key][0]

	for i in range(1,len(average[key])):
		tbl = [x + y for x, y in zip(tbl, average[key][i])]
	
	average[key] = [x/len(average[key]) for x in tbl]

for key in average:
	label.append(key)
	graphs.append(average[key])
	#print("gene values: ",list(seq_values[data_id]))
	#input("dance break)")

#Sort by biggest difference between all 4? (HOW?)

"""graphs=[
	[1,1,1,4,4,4,3,5,6,0],
	[1,1,1,5,5,5,3,5,6,0],
	[1,1,1,0,0,3,3,2,4,0],
	[1,2,4,4,3,2,3,2,4,0],
	[1,2,3,3,4,4,3,2,6,0],
	[1,1,3,3,0,3,3,5,4,3],
]"""

# plotting a line plot after changing it's width and height 
f = plt.figure() 
f.set_figwidth(2^16) 
f.set_figheight(2^9) 

for gg,graph in enumerate(graphs):
	plt.plot(grid,graph,label=list(average.keys())[gg])

plt.legend(loc=3,bbox_to_anchor=(1,0))
#plt.show()

plt.savefig("show3.png")

