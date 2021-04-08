import pandas as pd
import numpy as np

datas = pd.read_csv("master.csv", sep=';')
matrix = datas.to_numpy()

print(datas.head)
print(matrix[:10])

is_expressed = np.apply_along_axis(lambda x: np.mean(x == 0.0000) < .05, arr=matrix, axis=0)
print(is_expressed[:10])
print(print(np.size(is_expressed)))

l = []
for i in range(0,np.size(is_expressed) - 2):
    if is_expressed[i] == False: 
        l.append(i)
        
datas.drop(columns = datas.columns[l],
        inplace = True)

print(datas.shape)
print(datas.head)

datas.to_csv("matricen05.csv", index=False)