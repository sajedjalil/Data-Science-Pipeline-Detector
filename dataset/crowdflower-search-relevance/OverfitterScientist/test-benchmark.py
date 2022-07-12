
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import pipeline, metrics, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn import  pipeline#, metrics, grid_search,decomposition,
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
import string
from sklearn.feature_extraction import text


# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
stemmer = PorterStemmer()
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height',
'0','1','2','3','4','5','6','7','8','9']
#stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

#stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

punct = string.punctuation
punct_re = re.compile('[{}]'.format(re.escape(punct)))

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
stemmer = PorterStemmer()


train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

# we dont need ID columns
idx = test.id.values.astype(int)

for i in range(len(train.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
for i in range(len(test.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    t_data.append(s)

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
('rf', RandomForestClassifier())])


param_grid = {'rf__n_estimators': [160],'svd__n_components': [300],'rf__max_depth':[75]}

model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=25)
                                 
# Fit Grid Search Model
model.fit(s_data, s_labels)
preds = model.predict(t_data)
submission = pd.DataFrame({"id": idx, "prediction": preds})
submission.to_csv("ensemble_v1.csv", index=False)