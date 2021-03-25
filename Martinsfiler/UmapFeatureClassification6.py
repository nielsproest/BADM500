import pandas as pd
import numpy as np
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='poster')

datas = pd.read_csv("mDF05.csv", sep=',')
metadata = pd.read_csv("Classes05.csv")
matrix = datas.to_numpy()
y = np.ravel(metadata.to_numpy())

print("data head: ", matrix)
print("y: ", y)

umaps = UMAP(random_state=999, n_neighbors=30, min_dist=.25).fit_transform(matrix)
embedding = pd.DataFrame(umaps, columns = ['UMAP1','UMAP2'])
print("EM : ", embedding)
sns_plot = sns.scatterplot(x=embedding['UMAP1'], y=embedding['UMAP2'], data=embedding,
                hue=y, palette=sns.color_palette("husl", 5))
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
sns_plot.figure.savefig('umap_scatter_reduced_05.png', bbox_inches='tight', dpi=500)

