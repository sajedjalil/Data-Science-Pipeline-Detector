import pandas as pd
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import Ridge

train_data = pd.read_csv('../input/tabular-playground-series-jan-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-jan-2021/test.csv')
X_train = train_data.iloc[:, 1:-1].to_numpy()
X_test = test_data.iloc[:, 1:].to_numpy()
X = np.vstack([X_train, X_test])
y = train_data['target'].to_numpy()

embedder = RandomTreesEmbedding(
    n_estimators=800,
    max_depth=7,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
).fit(X)

X_train = embedder.transform(X_train)
X_test = embedder.transform(X_test)
model = Ridge(alpha=3000).fit(X_train, y)

sub = pd.read_csv('../input/tabular-playground-series-jan-2021/sample_submission.csv')
sub['target'] = model.predict(X_test)
sub.to_csv('submission.csv', index=False)