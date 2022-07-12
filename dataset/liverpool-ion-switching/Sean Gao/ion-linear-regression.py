import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

i = pd.read_csv('../input/liverpool-ion-switching/train.csv')

i = i[['signal', 'open_channels']]

X = i['signal']
X = np.array(X)
X = X.reshape(-1, 1)

y = i['open_channels']
y = y.values.ravel()

reg = LinearRegression().fit(X, y)

t = pd.read_csv('../input/liverpool-ion-switching/test.csv')

X_test = t['signal']
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 1)

y_pred = reg.predict(X_test)
y_pred = [round(x) for x in y_pred]

pred = t[['time']]
pred['open_channels'] = pd.DataFrame(y_pred)
pred['open_channels'] = pred['open_channels'].clip(lower = 0)
pred['open_channels'] = pred['open_channels'].abs()

pd.DataFrame.to_csv(pred, 'submission.csv',index=False, float_format='%.4f')