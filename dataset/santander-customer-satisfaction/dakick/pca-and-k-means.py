# a variation of the script taken from here:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
import pprint
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from time import time

droplist =  ['imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3',
                     'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'ind_var2', 'ind_var27', 'ind_var27_0',
                     'ind_var28', 'ind_var28_0', 'ind_var2_0', 'ind_var41', 'ind_var46', 'ind_var46_0',
                     'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
                     'num_trasp_var33_out_hace3', 'num_var27', 'num_var27_0', 'num_var28', 'num_var28_0',
                     'num_var2_0_ult1', 'num_var2_ult1', 'num_var41', 'num_var46', 'num_var46_0',
                     'saldo_medio_var13_medio_hace3', 'saldo_var27', 'saldo_var28', 'saldo_var2_ult1', 'saldo_var41',
                     'saldo_var46', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3',
                     'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3',
                     'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3', 'ind_var13_medio_0',
                     'ind_var18_0', 'ind_var25_0', 'ind_var26_0', 'ind_var6', 'ind_var6_0', 'ind_var32_0',
                     'ind_var34_0', 'ind_var37_0', 'ind_var40', 'num_var13_medio_0', 'num_var18_0', 'num_var25_0',
                     'num_var26_0', 'num_var6', 'num_var6_0', 'num_var32_0', 'num_var34_0', 'num_var37_0',
                     'num_var40', 'saldo_var13_medio', 'saldo_var6']


# ## Preprocessing

train = pd.read_csv('../input/train.csv', index_col=0)

X_train = train.drop(droplist+['TARGET'], axis=1)
Y_train = train['TARGET']
X_scaled = StandardScaler().fit_transform(X_train)
del(train)

pca_fit = PCA().fit(X_scaled)
pca_features = pca_fit.transform(X_scaled)
pca_len = (pd.Series(pca_fit.explained_variance_ratio_) >= .05).astype('int').sum()

pca_cols = ['PCA{:0>2d}'.format(i + 1) for i in range(pca_len)]

X_pca = pd.DataFrame(pca_features[:, :pca_len],
                                columns=pca_cols, index=X_train.index)
                                
del(X_train)

reduced_data_0 = np.array(X_pca.ix[Y_train  == 0, :])
reduced_data_1 = np.array(X_pca.ix[Y_train  == 1, :])
X_pca.head()
# ## Fitting K-Means
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=30)
kmeans.fit(reduced_data_0)
reduced_data = np.array(X_pca)

kmeans
# ## Original can be found here [Link](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py)
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(16,10))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data_0[:, 0], reduced_data_0[:, 1], 'k.', markersize=7)

plt.scatter(reduced_data_1[:, 0], reduced_data_1[:, 1],
            marker='o', s=50, linewidths=1,
            color='g', zorder=10)

# Plot the centroids as a white X


centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=20)



plt.title('K-means clustering on the Sandander dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()