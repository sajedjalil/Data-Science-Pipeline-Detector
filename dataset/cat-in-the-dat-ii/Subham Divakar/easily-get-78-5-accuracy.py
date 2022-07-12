# %% [code]
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce

df = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
df_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

# dropping unnecessary columns
test = df_test.drop(['id'],axis=1)
X = df.drop(['target','id'],axis=1)
y = pd.DataFrame(df['target'].copy())

# saving test ids for submission
test_id = pd.DataFrame(df_test['id'].copy())

# labeling all columns so thats why
columns_ = X.columns.tolist()
X.sort_index(inplace=True)

# encoded df
encoded = pd.DataFrame([])

# stratified k folds with target encoding
# normally 5-6 folds are enough
for tr_in,fold_in in StratifiedKFold(n_splits=5,shuffle=True).split(X,y):
    encoder = ce.TargetEncoder(cols = columns_,smoothing=0.2)
    encoder.fit(X.iloc[tr_in,:],y.iloc[tr_in])
    encoded = encoded.append(encoder.transform(X.iloc[fold_in,:]),ignore_index=False)
    
# for test data
encoder = ce.TargetEncoder(cols = columns_,smoothing=0.2)
encoder.fit(X,y)
test_ = encoder.transform(test)

#indexes are shuffled so sorting them
X_ = encoded.sort_index()

#splitting to check accuracy
X_train,X_test,y_train,y_test = train_test_split(X_,y)

# using logistic regression - very simple one!
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
# accuracy is: 0.8270666666666666

# final submission time
# fitting on whole training data
model.fit(X_,y)
y_hat = model.predict(test_)
res = pd.DataFrame(test_id)
# we need to give the prediction probability so that why
p = model.predict_proba(test_)
res['target'] = p[:,1]

# submission
res.to_csv('submission5.csv',index=False)