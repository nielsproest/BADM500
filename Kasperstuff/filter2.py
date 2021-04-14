import numpy as np
import pandas as pd

#Load data
print("Load data")
df = pd.read_csv("filterZeroes.csv",sep=';') # Run filter2.py first

# log(x+1) of every element in df. The iloc[] thing is to exclude sampleID and PAM50_Label
logged = np.log(df.iloc[:,1:-1]+1)

variance = logged.var().sort_values(ascending=False)

# taking the indeces from top17807 and reinserting ID's into the index
varianceIndexed = variance[:17807].index.insert(0, "sampleID")
# Adding the Pam50_label index back in as well
varianceIndexed = varianceIndexed.append(pd.Index(["PAM50_Label"]))
# filtering the data by the genes in the top 17807. AKA removing all genes with variance <0.2
data = df.filter(varianceIndexed)
print(varianceIndexed)
print(data)

df = pd.DataFrame(np.array(data), columns=varianceIndexed)
df.to_csv("masterNew3.csv", index=False, sep=";")