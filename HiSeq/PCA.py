import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import json

df = pd.read_csv("master.csv",sep=";")

types = df.columns.values
genes = types[1:-1] #Remove SampleID and PAM50_Label
X = df[genes]

pca = PCA(n_components=2)
components = pca.fit_transform(X)

samples = list(df["sampleID"])
fig = px.scatter(components, x=0, y=1, color=df["PAM50_Label"], hover_name=samples)

#TCGA-E9-A1N9-01
#TCGA-AO-A03R-01
#TCGA-AO-A03T-01
#TCGA-AO-A0JC-01

with open("outliers_1144.json") as f:
	dum = json.load(f)

	howmany = 15
	for i in dum[:howmany]:
		who = i[0]
		x,y = components[samples.index(who)]
		fig.add_annotation(x=x, y=y,
			text=who,
			showarrow=True,
			arrowhead=1
		)



fig.show()
