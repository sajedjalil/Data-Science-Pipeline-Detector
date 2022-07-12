# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
#https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print_every = 50
init_size = 2000
batch_size = 4000

def drop_empty_rows(df):
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    

def levenshtein_distance(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(t)]    
        
def find_optimal_clusters(data, max_k,column,init_size,batch_size):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=init_size, batch_size=batch_size, random_state=20).fit(data).inertia_)
        if (k % print_every == 0):
            print('Fit {} clusters for  column: {}'.format(k,column))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()
    
def plot_tsne_pca(data, labels,column):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=min(3000,data.shape[0]), replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=100).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot ' + column)
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot ' + column)
    plt.show()
    
def get_top_keywords(data, clusters, labels, n_terms,column):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {} column: {}'.format(i,column))
        print(','.join(set([labels[t] for t in np.argsort(r)[-n_terms:]])))

            
tfidf = TfidfVectorizer(
    min_df = 1,
    max_df = 0.95,
    stop_words = 'english',    
    max_features = 450
)

import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv',encoding='utf-8')
train["text"] = train["text"].astype(str)
train["text"] = train["text"].str.lower()
train["selected_text"] = train["selected_text"].astype(str)
train["selected_text"] = train["selected_text"].str.lower()
train["sentiment"] = train["sentiment"].astype(str)
train["sentiment"] = train["sentiment"].str.lower()
drop_empty_rows(train)

train, validation = train_test_split(train, train_size = 0.75, random_state = 0)


test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv',encoding='utf-8')
test["text"] = test["text"].astype(str)
test["text"] = test["text"].str.lower()
test["sentiment"] = test["sentiment"].astype(str)
test["sentiment"] = test["sentiment"].str.lower()
drop_empty_rows(test)

common_cols = list(set.intersection(*(set(df.columns) for df in [train,test])))
combined = pd.concat([df[common_cols] for df in [train,test]], ignore_index=True)
# applying groupby() function to 
# group the data on team value. 
gp = combined.groupby('sentiment') 
  
# Let's print the first entries 
# in all the groups formed. 
for name, group in gp: 
    print(name) 
    print(group) 
    print(len(group)) 

optimal_clusters = 100
for name, group in gp: 
    tfidf.fit(group.text)
    text = tfidf.transform(group.text)   
    find_optimal_clusters(text, optimal_clusters,name,init_size,batch_size)

kmeans_collection = {}
n_clusters = 100           
for name, group in gp: 
    tfidf.fit(group.text)
    text = tfidf.transform(group.text) 
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init_size=init_size, batch_size=batch_size, random_state=20)
    kmeans.fit(text)
    clusters = kmeans.predict(text) 
    plot_tsne_pca(text, clusters,name)  
    get_top_keywords(text, clusters, tfidf.get_feature_names(), 5,name)
    kmeans_collection[name.lower()] = kmeans


def get_keywords(line,data, clusters, labels, n_terms,column):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    selected_text = []
    for i,r in df.iterrows():
        #print('\nCluster {} column: {}'.format(i,column))
        #key_words = ','.join(set([labels[t] for t in np.argsort(r)[-n_terms:]]))
        #print(key_words)
        n_terms = len(labels)
        key_words = ','.join(set([labels[t] for t in np.argsort(r)[-n_terms:]]))
        for word in line.strip().split():
            for kw in key_words:
                word = word.strip()
                kw = kw.strip()
                ld = 1.0-levenshtein_distance(word,kw)/max(len(word),len(kw))
                if ld > 0.1:
                    selected_text.append(word) 
                    break
            #if word in key_words:
            #    selected_text.append(word)
    return " ".join(selected_text)

            
def jaccard(str1, str2): 
    if str1 and str2:
        a = set(str1.strip().split()) 
        b = set(str2.strip().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    else:
        return 0.0

scores = pd.DataFrame(columns = ["sentiment","text","selected_text","result","jaccard_score"])
count = 1
max_count = len(train)
print_every = 1000
gp = train.groupby('sentiment') 
for name, group in gp:
    for query,selected_text in zip(group.text,group.selected_text):
        text = tfidf.transform([query])      
        cluster = kmeans_collection[name.lower()].predict(text)
        result = get_keywords(selected_text,text,cluster,tfidf.get_feature_names(), 10,name)
        js = jaccard(selected_text,result)
        new_row = {'sentiment':name,'text':query, 'selected_text':selected_text, 'result':result, 'jaccard_score':js}
        scores = scores.append(new_row, ignore_index=True)
        if (count % print_every == 0):
            print("Train Processed:",count)
        count = count + 1
        if max_count < count:
            break
plt.figure()

'''
count_val = 1
scores_val = pd.DataFrame(columns = ["jaccard_score"])
for index in range(len(validation)):
    selected_text_val = tfidf.transform([validation.iloc[index]['selected_text']]) 
    text = tfidf.transform([validation.iloc[index]['text']])      
    cluster = kmeans_collection[validation.iloc[index]['sentiment'].lower()].predict(text)
    result_val = get_keywords(validation.iloc[index]['text'],text,cluster,tfidf.get_feature_names(), 10,name)
    js_new = jaccard(selected_text_val,result_val)
    new_row_val = {'jaccard_score':js_new}
    scores_val = scores_val.append(new_row_val, ignore_index=True)
    if (count_val % print_every == 0):
        print("Validation Processed:",count_val)
    count_val = count_val + 1
'''
scores.sort_values(by=['jaccard_score'],inplace=True,ascending=True)    
scores["jaccard_score"].plot.kde()
plt.hist(scores["jaccard_score"], color = 'blue', edgecolor = 'black')
plt.show()
print(scores["jaccard_score"].mean())

sample_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv',encoding='utf-8')
sample_submission["selected_text"] = sample_submission["selected_text"].astype(str)
for index in range(len(test)):
    text = tfidf.transform([test.iloc[index]['text']])      
    cluster = kmeans_collection[test.iloc[index]['sentiment'].lower()].predict(text)
    result = get_keywords(test.iloc[index]['text'],text,cluster,tfidf.get_feature_names(), 10,name)
    sample_submission.at[index,'selected_text'] = result
    if (index % print_every == 0):
        print("Result:" ,result)
        print("Test Processed:",index)
        
sample_submission.to_csv("submission.csv",index=False)
