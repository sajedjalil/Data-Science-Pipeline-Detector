# %% [markdown] {"editable":false}
# # Imports

# %% [code] {"_kg_hide-output":true,"editable":false,"execution":{"iopub.status.busy":"2021-11-25T01:48:38.085645Z","iopub.execute_input":"2021-11-25T01:48:38.085904Z","iopub.status.idle":"2021-11-25T01:48:41.901945Z","shell.execute_reply.started":"2021-11-25T01:48:38.085834Z","shell.execute_reply":"2021-11-25T01:48:41.901207Z"}}
import os
import re
import time
import warnings

# Data Manipulation Libraries
import pandas as pd
import numpy as np

# Data Vizualization libraries
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# %% [code] {"editable":false}
red = ["#4f000b","#720026","#ce4257","#ff7f51","#ff9b54"]
bo = ["#6930c3","#5e60ce","#0096c7","#48cae4","#ade8f4","#ff7f51","#ff9b54","#ffbf69"]
pink = ["#aa4465","#dd2d4a","#f26a8d","#f49cbb","#ffcbf2","#e2afff","#ff86c8","#ffa3a5","#ffbf81","#e9b827","#f9e576"]

# %% [markdown] {"editable":false}
# # Reading the data

# %% [code] {"editable":false}
#reading csv file
survey_df =  pd.read_csv("../input/kaggle-survey-2021/kaggle_survey_2021_responses.csv")
survey_df.head(6)

# %% [markdown] {"editable":false}
# # Questions Asked

# %% [code] {"_kg_hide-input":true,"editable":false}
fig = go.Figure(
    data=[
        go.Table(
        header=dict(
                values=["Question Number / Sections / Parts", "Description"],
                fill_color=bo[2],
                line_color='white',
                align='center'
        ),
        cells=dict(
                values=[
                    [i.replace('_'," ") for i in survey_df.columns[1:]],
                    survey_df.iloc[0,1:]
                ],
                fill_color=bo[4],
                line_color='white',
                align='left'
            )
        )
    ]
)
fig.update_layout(
    title = dict(
        text = 'Questions Asked in Survey 2021',
        font_size = 25,
    ),
    title_x=0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.show()

# %% [markdown] {"editable":false}
# # Data Preprocessing

# %% [code] {"editable":false}
interested_cols = [
    'Q1','Q2','Q3','Q4','Q5','Q6',
    'Q15',
    'Q20','Q21','Q22','Q23',
    'Q25'
]
df = survey_df.iloc[1:,1:][interested_cols]
df.columns = [
    'age','gender','country','edubg','profression','yrofexpc',
    'yrofexpml',
    'empindustry', 'companysize', 'empdsw', 'empdiml',
    'compensation'
]
print("Size of DataFrame: ", df.shape)
df.head()

# %% [markdown]
# Only found the above 12 columns looks quite decent which may add some meaning to identified clusters. All the above fields are categorical. I haven't included the multiple choice questions yet as I wasn't really sure about those and including didn't make any sense.

# %% [markdown] {"editable":false}
# # Dimensionality Reduction with TSNE and PCA

# %% [markdown]
# - Since all the features selected are categorical visualizing directly would yield much, it will look like a grid with different color balls placed at the intersections. So now lets reduce the number of dimensions to 2 components: it will be easy visualize, will take less time to train, and find the why more number of components are required.
# 
# - PCA is usually used to reduce the dimension of datasets which are very large, whereas TSNE is used to reduce the dimensions of the dataset which has a very large number of features. Although, we are aware about this will still give it a try and find what comes up.
# 
# #### Let's START

# %% [code] {"editable":false}
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding

le = LabelEncoder()

for col in df.columns:
    df[col] = df[col].fillna('Unknown')
    df[col] = le.fit_transform(df[col])

pipe4 = Pipeline(
    [
        ('scaler', StandardScaler()), 
#         ('tsne', TSNE(n_components=2, verbose=1, perplexity=46, n_iter=550)),
        ('SE', SpectralEmbedding(2))
    ]
)
df_se = pd.DataFrame(pipe4.fit_transform(df))
# df_pca.head()
    
# LocallyLinearEmbedding
pipe3 = Pipeline(
    [
        ('scaler', StandardScaler()), 
#         ('tsne', TSNE(n_components=2, verbose=1, perplexity=46, n_iter=550)),
        ('LLE', LocallyLinearEmbedding(2))
    ]
)
df_lle = pd.DataFrame(pipe3.fit_transform(df))
# df_pca.head()
    
# PCA
#pipe2 = Pipeline(
#    [
#        ('scaler', StandardScaler()), 
#         ('tsne', TSNE(n_components=2, verbose=1, perplexity=46, n_iter=550)),
#        ('pca', PCA(2))
#    ]
#)
#df_pca = pd.DataFrame(pipe2.fit_transform(df))
#df_pca.head()

# TSNE
#pipe = Pipeline(
#    [
#        ('scaler', StandardScaler()), 
#        ('tsne', TSNE(n_components=2, verbose=1, perplexity=46, n_iter=550)),
#         ('pca', PCA(2))
#    ]
#)
#df_r = pd.DataFrame(pipe.fit_transform(df))

#df_r.head()

# %% [code] {"editable":false}
fig, ax = plt.subplots(1,2,figsize=(20, 9))

# plt.figure(figsize=(10,8))
sns.scatterplot(
    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
    data=df,
    legend="full",
    ax=ax[0]
)
ax[0].set_title("PCA")

sns.scatterplot(
    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
    data=df,
    legend="full",
    ax=ax[1]
)
ax[1].set_title("TSNE")

# %% [markdown] {"editable":false}
# # Finding Optimal Number of Clusters

# %% [code]
from sklearn.cluster import KMeans

def no_of_cluster(df, title):
    w=[]
    e=[]
    for i in range(1,10):
        k=KMeans(n_clusters=i)
        k.fit_predict(df)
        e.append(k.inertia_)
        w.append(i)
    plt.figure(figsize=(8,5))
    plt.plot(w,e,'bo-')
    plt.title(f"Optimum number of Clusters for SpectralClustering - {title}")
no_of_cluster(df_se, "SE")
no_of_cluster(df_lle, "LLE")
no_of_cluster(df_pca, "PCA")
no_of_cluster(df_r, "TSNE")

# %% [markdown]
# To find the optimal number of clusters, let's have a look at the above elbow plots for PCA and TSNE. The points 3 and 4 looks good enough, but yet 3 seems to be a bit more stable.

# %% [markdown]
# # Training the Spectral Clustering Model

# %% [code]

model_se = KMeans(n_clusters=3, random_state=50)
model_pca.fit(df_se)

model_lle = KMeans(n_clusters=3, random_state=50)
model_pca.fit(df_lle)

#model_pca = KMeans(n_clusters=3, random_state=50)
#model_pca.fit(df_pca)

#model_tsne = KMeans(n_clusters=3, random_state=50)
#model_tsne.fit(df_r)

# %% [markdown] {"editable":false}
# # Visualizing the cluster mappings

# %% [code] {"editable":false}
fig, ax = plt.subplots(1,2,figsize=(20, 9))

ax[0].scatter(df_se.iloc[:,0], df_se.iloc[:,1], c=model_se.labels_, alpha=0.3)
ax[0].set_title("SE")

ax[0].scatter(df_lle.iloc[:,0], df_lle.iloc[:,1], c=model_lle.labels_, alpha=0.3)
ax[0].set_title("LLE")

#ax[0].scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], c=model_pca.labels_, alpha=0.3)
#ax[0].set_title("PCA")

#ax[1].scatter(df_r.iloc[:,0], df_r.iloc[:,1], c=model_tsne.labels_, alpha=0.3)
#ax[1].set_title("TSNE")

# %% [markdown] {"editable":false}
# # Comparing TSNE and PCA

# %% [markdown] {"editable":false}
# ## Profession based distribution

# %% [code] {"editable":false}
fig, ax = plt.subplots(2,2,figsize=(20, 18))

# SE
ax[0,0].scatter(df_se.iloc[:,0],df_se.iloc[:,1],c=model_se.labels_, alpha=0.3)
sns.scatterplot(
    x=df_se.iloc[:,0], y=df_se.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q5'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("SE")

# LLE
ax[0,0].scatter(df_lle.iloc[:,0],df_lle.iloc[:,1],c=model_lle.labels_, alpha=0.3)
sns.scatterplot(
    x=df_lle.iloc[:,0], y=df_lle.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q5'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("LLC")

# PCA
#ax[0,0].scatter(df_pca.iloc[:,0],df_pca.iloc[:,1],c=model_pca.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q5'],
#    data=df,
#    legend="full",
#    ax=ax[0,1]
#)
#ax[0,1].set_title("PCA")

# TSNE
#ax[1,0].scatter(df_r.iloc[:,0],df_r.iloc[:,1],c=model_tsne.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q5'],
#    data=df,
#    legend="full",
#    ax=ax[1,1]
#)
#ax[1,1].set_title("TSNE")

# Plot legend outside of plot
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %% [markdown] {"editable":false}
# ## Gender based Distribution

# %% [code] {"editable":false}
fig, ax = plt.subplots(2,2,figsize=(20, 18))

# SE
ax[0,0].scatter(df_se.iloc[:,0],df_se.iloc[:,1],c=model_se.labels_, alpha=0.3)
sns.scatterplot(
    x=df_se.iloc[:,0], y=df_se.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q2'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("SE")

# LLE
ax[0,0].scatter(df_lle.iloc[:,0],df_lle.iloc[:,1],c=model_lle.labels_, alpha=0.3)
sns.scatterplot(
    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q2'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("LLE")

# PCA
#sns.scatterplot(
#    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q2'],
#    data=df,
#    legend="full",
#    ax=ax[0,1]
#)
#ax[0,1].set_title("PCA")

# TSNE
#ax[1,0].scatter(df_r.iloc[:,0],df_r.iloc[:,1],c=model_tsne.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q2'],
#    data=df,
#    legend="full",
#    ax=ax[1,1]
#)
#ax[1,1].set_title("TSNE")

# Plot legend outside of plot
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %% [markdown] {"editable":false}
# ## Coding Experience

# %% [code] {"editable":false}
fig, ax = plt.subplots(2,2,figsize=(20, 18))

# SE
ax[0,0].scatter(df_se.iloc[:,0],df_se.iloc[:,1],c=model_se.labels_, alpha=0.3)
sns.scatterplot(
    x=df_se.iloc[:,0], y=df_se.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q6'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("SE")

# LLE
ax[0,0].scatter(df_lle.iloc[:,0],df_lle.iloc[:,1],c=model_lle.labels_, alpha=0.3)
sns.scatterplot(
    x=df_lle.iloc[:,0], y=df_lle.iloc[:,1],
    hue=survey_df.iloc[1:,1:]['Q6'],
    data=df,
    legend="full",
    ax=ax[0,1]
)
ax[0,1].set_title("LLE")

#ax[0,0].scatter(df_pca.iloc[:,0],df_pca.iloc[:,1],c=model_pca.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q6'],
#    data=df,
#    legend="full",
#    ax=ax[0,1]
#)
#ax[0,1].set_title("PCA")

# TSNE
#ax[1,0].scatter(df_r.iloc[:,0],df_r.iloc[:,1],c=model_tsne.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q6'],
#    data=df,
#    legend="full",
#    ax=ax[1,1]
#)
#ax[1,1].set_title("TSNE")

# Plot legend outside of plot
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %% [markdown] {"editable":false}
# ## Experience in Machine Learning

# %% [code] {"editable":false}
fig, ax = plt.subplots(2,2,figsize=(20, 18))

# PCA
#ax[0,0].scatter(df_pca.iloc[:,0],df_pca.iloc[:,1],c=model_pca.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q15'],
#    data=df,
#    legend="full",
#    ax=ax[0,1]
#)
#ax[0,1].set_title("PCA")

# TSNE
#ax[1,0].scatter(df_r.iloc[:,0],df_r.iloc[:,1],c=model_tsne.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q15'],
#    data=df,
#    legend="full",
#    ax=ax[1,1]
#)
#ax[1,1].set_title("TSNE")

# Plot legend outside of plot
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %% [markdown] {"editable":false}
# ## Compensation

# %% [code] {"editable":false}
fig, ax = plt.subplots(2,2,figsize=(20, 18))

# PCA
#ax[0,0].scatter(df_pca.iloc[:,0],df_pca.iloc[:,1],c=model_pca.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_pca.iloc[:,0], y=df_pca.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q25'],
#    data=df,
#    legend="full",
#    ax=ax[0,1]
#)
#ax[0,1].set_title("PCA")

# TSNE
#ax[1,0].scatter(df_r.iloc[:,0],df_r.iloc[:,1],c=model_tsne.labels_, alpha=0.3)
#sns.scatterplot(
#    x=df_r.iloc[:,0], y=df_r.iloc[:,1],
#    hue=survey_df.iloc[1:,1:]['Q25'],
#    data=df,
#    legend="full",
#    ax=ax[1,1]
#)
#ax[1,1].set_title("TSNE")

# Plot legend outside of plot
ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %% [markdown]
# ### Seems like 2 dimension aren't enough to carve out good difference between clusters.
# - It is all mixed up and cluttered
# - Lets now try with increasing the number of components to 3 or 4 and observe the differences.

# %% [markdown]
# # Reducing Dimensionality to 3 components

# %% [code]
# PCA
#pipe24 = Pipeline(
#    [
#        ('scaler', StandardScaler()), 
#         ('tsne', TSNE(n_components=2, verbose=1, perplexity=46, n_iter=550)),
#        ('pca', PCA(3))
#    ]
#)
# df4_pca = pd.DataFrame(pipe24.fit_transform(df))
# df_pca.head()

# TSNE
# pipet = Pipeline(
#     [
#         ('scaler', StandardScaler()), 
#         ('tsne', TSNE(n_components=3, verbose=1, perplexity=104, n_iter=550)),
# #         ('pca', PCA(2))
#     ]
# )
# df4_r = pd.DataFrame(pipet.fit_transform(df))

# %% [code]
#fig, ax = plt.subplots(3,3,figsize=(20, 9))
#fig.suptitle('PCA components')

#for i in range(3):
#    for j in range(3):
#        if i!=j:
#            sns.scatterplot(
#                x=df4_pca.iloc[:,i], y=df4_pca.iloc[:,j],
#                data=df,
#                legend="full",
#                ax=ax[i,j]
#            )
#            ax[i,j].set_title(f"{i} vs {j}")

# %% [code]
# no_of_cluster(df4_pca, "PCA")

# %% [code]
# model4_pca = KMeans(n_clusters=3, random_state=70)
# model4_pca.fit(df4_pca)

# %% [code]
#fig = plt.figure(figsize = (10, 10))
#ax = plt.axes(projection ="3d")

#ax.scatter3D(
#    df4_pca.iloc[:,0], 
#    df4_pca.iloc[:,1], 
#    df4_pca.iloc[:,2], 
#    c=model4_pca.labels_, 
#    alpha=0.3
#)
#plt.title("Visualizing clusters in 3D")

# %% [code]
#fig = plt.figure(figsize = (10, 10))
#ax = plt.axes(projection ="3d")
#ax.scatter3D(
#    df4_pca.iloc[:,0], 
#    df4_pca.iloc[:,1], 
#    df4_pca.iloc[:,2], 
#    c=le.fit_transform(survey_df.fillna('Unknown').iloc[1:,1:]['Q25']), 
#    alpha=0.3
#)

# %% [markdown] {"editable":false}
# # To be continued...
# 
# #### Upvote if you find it interesting. Do not forget to drop suggestions in comments below.