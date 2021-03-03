import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')

zerosarr = []

# Adding the amount of zeroes from each gene into a list
for column in df.columns[1:]:
    y = df[column].to_numpy()
    num_zeros = (y == 0).sum()
    zerosarr.append(num_zeros)

zerosarr.sort()
# Same list just in percentages instead of amounts of zeroes, not used
zerospct = [(x/1144)*100 for x in zerosarr]

plt.plot(zerosarr)
# x-axis label 
plt.xlabel('genes') 
# frequency label 
plt.ylabel('# of zeroes') 

# Arrows
plt.annotate('27 zeroes, or 2.4%', xy=(15000,28), xytext=(8000,110),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('10% of genes have >857 zero-values', xy=(18478,857), xytext=(2000,600),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('More than 1400 genes (7%) have >90% zero-values', xy=(19100,1032), xytext=(1000,1000),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig("test.png")
