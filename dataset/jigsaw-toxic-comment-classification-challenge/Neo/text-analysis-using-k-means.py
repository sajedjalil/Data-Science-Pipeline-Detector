import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

doc_train = train['comment_text']
doc_test = test['comment_text']
all_text = pd.concat([doc_train, doc_test])

#cleaned word
def letters_only(astr):
    for c in astr:
        if not c.isalpha():
            return False

    return True

cv = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',stop_words='english',max_features=10000)
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in all_text:
    cleaned.append(' '.join([lemmatizer.lemmatize(word.lower())
                             for word in post.split()
                             if letters_only(word)
                             and word not in all_names]))

transformed = cv.fit_transform(cleaned)

km = KMeans(n_clusters=20)
km.fit(transformed)

labels = len(all_text)
plt.scatter(labels, km.labels_)
plt.xlabel('Data')
plt.ylabel('Cluster')
plt.show()


