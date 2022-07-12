import numpy as np 
import seaborn as sns
import matplotlib
import pandas as pd 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {"axes.facecolor": ".95"})

data = pd.read_csv('../input/train.csv')

species = data['species']
data = data.drop(['species','id'],axis = 1)

tsne = TSNE()
X_tsne = tsne.fit_transform(np.array(data))

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
proj = pd.DataFrame(X_tsne)
proj.columns = ["TSNE_1", "TSNE_2"]
proj["labels"] = species
sns.lmplot("TSNE_1", "TSNE_2", hue = "labels", data = proj ,fit_reg=False, legend=False)
# remove legend due to output rendering (try locally to see the results)


plt.xlabel('TSNE_1')
plt.ylabel('TSNE_2')
plt.title('T-SNE')
plt.savefig('1.png')
plt.show()  
