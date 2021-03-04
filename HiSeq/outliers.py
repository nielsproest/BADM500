import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from outlib import mad_based_outlier, percentile_based_outlier, plot
#from scipy.stats import median_absolute_deviation

import json, collections

dt = pd.read_csv("master.csv",sep=";")

types = dt.columns.values
genes = types[1:-1]
samples = list(dt["sampleID"])

j=0
#print(genes)
outlier = {}
for i in genes:
	vals = np.array(dt[i])
	outliers = percentile_based_outlier(vals)
	#input(outliers)
	for k, v in enumerate(outliers):
		if v:
			who = samples[k] #headers are ignored
			#print(who)
			if not who in outlier:
				outlier[who] = 1
			else:
				outlier[who] = outlier[who] + 1
	#print(i)

	plot(vals)
	j=j+1
	if (j>=4):
		plt.show()

	#input(vals, outliers)

data = [[i, outlier[i]] for i in outlier]
data.sort(key=lambda l: l[1])
data.reverse()
with open("outliers_{}.json".format(str(len(data))), "w") as f:
	json.dump(data, f, indent="\t")
