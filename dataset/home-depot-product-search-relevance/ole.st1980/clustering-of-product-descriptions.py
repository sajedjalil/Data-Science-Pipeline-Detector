#/usr/bin/python

# A cursory look at the content of product_description.csv from
# Kaggle's home depot product challenge
#
# I should say clearly that I don't know what I'm doing.
# I've wrote this script on a rainy Saturday afternoon to poke
# poke around scikit-learn. Any comments appreciated.

from __future__ import print_function
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import random

# This can be improved a lot
def preprocessor(data):
    return " ".join([SnowballStemmer("english").stem(word) for word in data.split()])

pdesc = pd.read_csv('../input/product_descriptions.csv')

nrows = pdesc.shape[0]
# change the random seed to get a different subsample of the data, see below
random.seed(42)

nsamples = 1000 # we work only with a subset of the data
# Number of dimension we project the tf-idf counts to
ndim = 3
# number of clusters for k-means. Just a guess
num_clusters = 6

# take random sample
rows = random.sample(list(pdesc.index), nsamples)
pdesc = pdesc.ix[rows]
print(pdesc.head())

# Tf-idf
# See http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
t0 = time.time()
desc_vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                  #max_df=0.8,
                                  #min_df=0.01,
                                  stop_words='english')

desc_vectorizer = desc_vectorizer.fit(pdesc['product_description'])
pdesc_transformed = desc_vectorizer.fit_transform(pdesc['product_description'])
print("Tf-idf in %fs" % (time.time() - t0))

# Cluster on Tf-idf output using K-means
t0 = time.time()
km = KMeans(n_clusters=num_clusters, 
            init='k-means++', 
            max_iter=100, 
            n_init=1,
            verbose=True)
km.fit(pdesc_transformed)
print("K-means in %0.3fs" % (time.time() - t0))

print("Top word per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = desc_vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %s :" % i,end=" ")
    for idx in order_centroids[i, :5]:
        print("%s" % (terms[idx]),end=" ")
    print()

# Reduce dimensions using Singular Value Decomposition
t0 = time.time()
svd = TruncatedSVD(ndim)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
pdesc_transformed_svd = lsa.fit_transform(pdesc_transformed)
print("SVD in %fs" % (time.time() - t0))
explained_variance = svd.explained_variance_ratio_.sum()
# The svd explains only a tiny portion of the variance
print("Explained variance %fs" % (explained_variance))


# and plot reduced representation
# hard to draw any conclusions from this plot,
# apart from the fact that there is no obvious clustering
# in 3 dimensions
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pdesc_transformed_svd[:, 0], 
           pdesc_transformed_svd[:, 1],
           pdesc_transformed_svd[:, 2])

plt.savefig('clusters_pdesc.png', bbox_inches='tight')
