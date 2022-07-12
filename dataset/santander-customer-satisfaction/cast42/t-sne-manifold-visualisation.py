# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame

from sklearn import manifold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize



X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']
tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, method='barnes_hut', n_iter=1000, verbose=20)

features = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult3', 
'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 
'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_ult1', 'ind_var8_0', 
'ind_var30_0', 'ind_var30', 'num_op_var41_hace2', 'num_op_var41_ult3', 
'num_var37_med_ult2', 'saldo_var5', 'saldo_var8', 'saldo_var26', 'saldo_var30', 
'saldo_var37', 'saldo_var42', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 
'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3',
'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3',
'num_op_var39_comer_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1',
'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult3',
'saldo_medio_var13_corto_hace2', 'var38', 'n0']

X_sel = X[features]

X_norm = normalize(X_sel, axis=0)

# Select 10% from train vector to reduce memory requirements of t-sne
X_train, X_test, y_train, y_test = \
  train_test_split(X_norm, y, random_state=1301, stratify=y, test_size=0.1)

X_tsne = tsne.fit_transform(X_test)

plt.figure()
plt.scatter(X_tsne[np.where(y_test == 0), 0],
                   X_tsne[np.where(y_test == 0), 1],
                   marker='x', color='g',
                   linewidth='1', alpha=0.8, label='Happy')
plt.scatter(X_tsne[np.where(y_test == 1), 0],
                   X_tsne[np.where(y_test == 1), 1],
                   marker='v', color='r',
                   linewidth='1', alpha=0.8, label='Unhappy')

plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE on 10% train samples')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show()  
