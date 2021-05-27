import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import sys

print("Load")
dataset = pd.read_csv("../Kasperstuff/masterNew.csv", sep=";")

print("Mod")
dataset.index = dataset[dataset.columns.values[0]] #Fix row labels
dataset = dataset.drop(["sampleID","PAM50_Label"], axis=1)

#Calculate gene's, not samples (although that might be interesting too)
dataset = dataset.T

print("Work")
#sys.setrecursionlimit(1000000000) #Haha, cpu get owned
link = sch.linkage(dataset, method="average", metric="correlation")
dendrogram = sch.dendrogram(link, labels=dataset.index, truncate_mode="lastp", p=2500)
genes = [i for i in dendrogram["ivl"] if not i.startswith("(")]
print(genes) #327 for 1000
print(len(genes))

print("Save")
plt.title('Dendrogram')
plt.xlabel('Genes')
plt.ylabel('Correlation')
plt.show()
