import numpy as np
import pandas as pd

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')

# var() is variance
variance = df.var().sort_values(ascending=False)

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
df.to_csv("masterNew.csv", index=False, sep=";")