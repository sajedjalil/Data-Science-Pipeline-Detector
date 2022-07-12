import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')



df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]

vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
X = vectorizer.fit_transform(df)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))



loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    print('ROC AUC:', roc_auc_score(train[j], pred_train))
    loss.append(roc_auc_score(train[j], pred_train))
    
print('mean column-wise ROC AUC:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)