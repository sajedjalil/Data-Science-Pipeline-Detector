import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

data_path = '../input/'
train = pd.read_csv(data_path + 'train_2016.csv')
ss = pd.read_csv(data_path + 'sample_submission.csv')

subm = ss.copy()
n = 4

# Simple mean
mu = round(train.logerror.mean(), n)
ans = [mu] * len(train['logerror'])
errs = mean_absolute_error(train['logerror'].values, ans)
print('mu = ' + str(mu) + ', mae = ' + str(errs) + ', LB = 0.0652995')

# subm['201610'] = mu
# subm['201611'] = mu
# subm['201612'] = mu

# subm['201710'] = mu
# subm['201711'] = mu
# subm['201712'] = mu

# Brute force
start_val = -0.02
step = 0.0001
iters = 400

mu_vals = np.array([start_val + step*i for i in range(iters)])
errs = np.empty(len(mu_vals))
for i, mu in enumerate(mu_vals):
    ans = [mu] * len(train['logerror'])
    errs[i] = mean_absolute_error(train['logerror'].values, ans)
mu = mu_vals[np.argmin(errs)]
print('mu = ' + str(mu) + ', mae = ' + str(np.min(errs)) + ', LB = 0.0656439')

# subm['201610'] = mu
# subm['201611'] = mu
# subm['201612'] = mu

# subm['201710'] = mu
# subm['201711'] = mu
# subm['201712'] = mu

# Month-specific brute force
train['month'] = train.transactiondate.apply(lambda x: int(x.split('-')[1]))
mu_vals = np.array([start_val + step*i for i in range(iters)])
mu_buf = np.zeros(3)
for m_i, m in enumerate(range(10, 13)):
    errs = np.empty(len(mu_vals))
    for i, mu in enumerate(mu_vals):
        ans = [mu] * len(train[train['month'] >= m]['logerror'].values)
        errs[i] = mean_absolute_error(train[train['month'] >= m]['logerror'].values, ans)
    mu_buf[m_i] = mu_vals[np.argmin(errs)]
print('mu = ' + str(mu_buf) + ', LB = 0.0653038')

subm['201610'] = mu_buf[0]
subm['201611'] = mu_buf[1]
subm['201612'] = mu_buf[2]

subm['201710'] = mu_buf[2]
subm['201711'] = mu_buf[2]
subm['201712'] = mu_buf[2]

subm.to_csv('submission.csv', index=False, float_format=('%.' + str(n) + 'f'))
