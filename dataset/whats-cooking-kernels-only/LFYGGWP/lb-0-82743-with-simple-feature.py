# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json
import re
import unidecode
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from tqdm import tqdm
tqdm.pandas()

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

train['num_ingredients'] = train['ingredients'].apply(lambda x: len(x))
test['num_ingredients'] = test['ingredients'].apply(lambda x: len(x))
#train['num_ingredients_2'] = train['ingredients'].apply(lambda x: len(x)*len(x))
#train['num_ingredients_log'] = train['ingredients'].apply(lambda x: log(len(x))
train = train[train['num_ingredients'] > 2]


lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', '')#wasabe
    ingredients_text = ingredients_text.replace('wasabe', 'wasabi') #for wrong name
    #ingredients_text = ingredients_text.replace('egg whites', 'eggwhites , egg , whites')
    #ingredients_text = ingredients_text.replace('lime juice', 'limejuice')
    #ingredients_text = ingredients_text.replace('clam juice', 'clamjuice')
    #ingredients_text = ingredients_text.replace('lemon juice', 'lemonjuice')
    #ingredients_text = ingredients_text.replace('orange juice', 'orangejuice')
    #ingredients_text = ingredients_text.replace('soy sauce', 'soysauce')
    ingredients_text = ingredients_text.replace('fish sauce', 'fishsauce')
    #ingredients_text = ingredients_text.replace('sesame oil', 'sesameoil')
    #ingredients_text = ingredients_text.replace('olive oil', 'oliveoil')#vegetable oil corn oil
    #ingredients_text = ingredients_text.replace('vegetable oil', 'vegetableoil')
    #ingredients_text = ingredients_text.replace('corn oil', 'cornoil')#rice wine
    #
    ingredients_text = ingredients_text.replace('coconut cream', 'coconutcream')
    #
    ingredients_text = ingredients_text.replace('yellow onion', 'yellowonion')
    ingredients_text = ingredients_text.replace('cream cheese', 'creamcheese') 
    ingredients_text = ingredients_text.replace('baby spinach', 'babyspinach')
    ingredients_text = ingredients_text.replace('coriander seeds', 'corianderseeds')
    ingredients_text = ingredients_text.replace('corn tortillas', 'corntortillas')
    ingredients_text = ingredients_text.replace('rice cakes', 'ricecakes')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)
'''
for ingredient, expected in [
    ('eggs', 'egg'),
    ('all-purpose flour', 'all purpose flour'),
    ('purée', 'purée'),
    ('1% low-fat milk', 'lowfat milk'),
    ('half & half', 'half half'),
    ('safetida (powder)', 'safetida (powder)')
]:
    actual = preprocess([ingredient])
    assert actual == expected, f'"{expected}" is excpected but got "{actual}"'
'''
train['x'] = train['ingredients'].progress_apply(lambda ingredients: preprocess(ingredients))
test['x'] = test['ingredients'].progress_apply(lambda ingredients: preprocess(ingredients))
train.head()

vectorizer = make_pipeline(
    TfidfVectorizer(sublinear_tf=True),
    FunctionTransformer(lambda x: x.astype('float'), validate=False)
)

x_train = vectorizer.fit_transform(train['x'].values)
x_train.sort_indices()
x_test = vectorizer.transform(test['x'].values)

#x_train = x_train.toarray()
#x_test = x_test.toarray()
#x_train = np.concatenate((train['num_ingredients'], x_train), axis = 1)
#x_test = np.concatenate((test['num_ingredients'], x_test), axis = 1)

#x_train = np.column_stack((train['num_ingredients'], x_train))
#x_test = np.column_stack((test['num_ingredients'], x_test))

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train['cuisine'].values)
dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


estimator = SVC(C=250, # penalty parameter, setting it to a larger value 
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value, not tuned yet
	 			 gamma=1.4, # kernel coefficient, not tuned yet
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	 			 probability=False, # no need to enable probability estimates
	 			 cache_size=1000, # 200 MB cache size
	 			 class_weight=None, # all classes are treated equally 
	 			 verbose=False, # print the logs 
	 			 max_iter=-1, # no limit, let it run
	 			 decision_function_shape=None, # will use one vs rest explicitly 
	 			 random_state=None)
classifier = OneVsRestClassifier(estimator, n_jobs=-1)

scores = cross_validate(classifier, x_train, y_train, cv=3)
scores['test_score'].mean()

classifier.fit(x_train, y_train)

y_pred = label_encoder.inverse_transform(classifier.predict(x_train))
y_true = label_encoder.inverse_transform(y_train)

print(f'accuracy score on train data: {accuracy_score(y_true, y_pred)}')

def report2dict(cr):
    rows = []
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0: rows.append(parsed_row)
    measures = rows[0]
    classes = defaultdict(dict)
    for row in rows[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            classes[class_label][m.strip()] = float(row[j + 1].strip())
    return classes
report = classification_report(y_true, y_pred)
pd.DataFrame(report2dict(report)).T


y_pred = label_encoder.inverse_transform(classifier.predict(x_test))
test['cuisine'] = y_pred
test[['id', 'cuisine']].to_csv('submission.csv', index=False)
test[['id', 'cuisine']].head()


    


