import numpy as np
import pandas as pd

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')

'''
# var() is variance
variance = df.var().sort_values(ascending=False)
''''''
# https://en.wikipedia.org/wiki/Coefficient_of_variation
# A more relative way to look at variance. 
# For now we want to remove the same amount of genes, though we expect different genes to be removed with this method.
# The genes with ONLY zeroes return NaN because we divide with zero. Does not matter as they are put last when sorting and thrown out.
std = df.std()
mean = df.mean()
rsd = std/mean

variance = rsd.sort_values(ascending=False)
'''
# Remove ID's and PAM50 labels. Already wrote the code earlier to add them back in, so this makes the np.log below easier.
del df["PAM50_Label"]
del df["sampleID"]
# log(x+1) of every element in df
np.log(df+1)



# primitive way of finding the index where variance is <0.2 
# Result: 17807
'''
for row, data in enumerate(variance):
    if (data < 0.2):
        print(row)
        print(data)
        break
'''

# taking the indeces from top17807 and reinserting ID's into the index
varianceIndexed = variance[:17807].index.insert(0, "sampleID")
# Adding the Pam50_label index back in as well
varianceIndexed = varianceIndexed.append(pd.Index(["PAM50_Label"]))
print(variance)
# filtering the data by the genes in the top 17807. AKA removing all genes with variance <0.2
data = df.filter(varianceIndexed)


df = pd.DataFrame(np.array(data), columns=varianceIndexed)
df.to_csv("masterNew2.csv", index=False, sep=";")