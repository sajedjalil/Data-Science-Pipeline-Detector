import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Load data
train = pd.read_json("../input/train.json")
test  = pd.read_json("../input/test.json")
num_train, num_test = len(train), len(test)
print(num_train, num_test)

df = pd.concat([train, test]).set_index('id')
del train, test

# Preprocess
def tokenize(ingredients):
    for ingredient in ingredients:
        sub = ingredient.split()
        for w in sub:
            yield w
        if len(sub) > 1:
            yield ingredient

df['ingredients_str'] = df['ingredients'].apply(lambda x: ','.join(tokenize(x)))

# ---------------------- Transform -------------===-----------
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Get trainset and testset
train = df.iloc[:num_train]
test = df.iloc[num_train:]

# Label encoding
le = LabelEncoder()
y = le.fit_transform(train['cuisine'])

# TF-IDF
vectorizer = Tfidf(min_df=5, 
                   max_df=0.9,
                   tokenizer=lambda x: x.split(','),
                   sublinear_tf=True, 
                  )

X_train = vectorizer.fit_transform(train['ingredients_str'])
X_test = vectorizer.transform(test['ingredients_str'])
print(X_train.shape)

# Split and get validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.05, random_state=321)

# ---------------------------- Training ---------------------------------
import lightgbm as lgb
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclassova',
    # 'objective': 'multiclass',
    'num_class': 20,
    'metric': {'multi_error'},
    'num_leaves': 31,
    'learning_rate': 0.04,
}

gbm = lgb.train(params, 
                dtrain,
                num_boost_round=300,
                valid_sets=dvalid,
                verbose_eval=30,
                early_stopping_rounds=30)

# Predict
y_pred = gbm.predict(X_test, 
                     num_iteration=gbm.best_iteration).argmax(axis=1)
y_pred = le.inverse_transform(y_pred)

subm = test[['cuisine']].copy()
subm['cuisine'] = y_pred
subm.to_csv("submission.csv")