
# coding: utf-8

# The following notebook introduces ML-Ensemble, a Python library for memory-efficient parallel ensemble learning with a Scikit-learn API. 
# 
# ML-Ensemble also deploys a neural network-like API for building ensembles of several layers, and can accomodate a great variety of ensemble architectures. 
# 
# For more information, see http://mlens.readthedocs.io/en/latest/index.html.

# In[1]:


import gc
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Inputs
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Data viz
from mlens.visualization import corr_X_y, corrmat

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from mlens.preprocessing import EnsembleTransformer

# Ensemble
from mlens.ensemble import SuperLearner

from scipy.stats import uniform, randint

from matplotlib.pyplot import show



# In[2]:


SEED = 148
np.random.seed(SEED)


# # 1. Getting a good baseline for ensemble learning
# 
# It's always good to check how inputs play along with the output.
# Here, we highlight one example functionality of the Ml-Ensemble's
# visualization library.

# In[3]:


def build_train():
    """Read in training data and return input, output, columns tuple."""

    # This is a version of Anovas minimally prepared dataset
    # for the xgbstarter script
    # https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

    df = pd.read_csv('../input/train_2016.csv')

    prop = pd.read_csv('../input/properties_2016.csv')
    convert = prop.dtypes == 'float64'
    prop.loc[:, convert] =         prop.loc[:, convert].apply(lambda x: x.astype(np.float32))

    df = df.merge(prop, how='left', on='parcelid')

    y = df.logerror
    df = df.drop(['parcelid',
                  'logerror',
                  'transactiondate',
                  'propertyzoningdesc',
                  'taxdelinquencyflag',
                  'propertycountylandusecode'], axis=1)

    convert = df.dtypes == 'object'
    df.loc[:, convert] =         df.loc[:, convert].apply(lambda x: 1 * (x == True))

    df.fillna(0, inplace=True)

    return df, y, df.columns


# In[4]:


xtrain, ytrain, columns = build_train()
xtrain, xtest, ytrain, ytest = train_test_split(xtrain,
                                                ytrain,
                                                test_size=0.5,
                                                random_state=SEED)


# In[5]:


# this plot requires mlens 0.1.3, Kaggle is currently on 0.1.2
#corr_X_y(xtrain, ytrain, figsize=(16, 10), label_rotation=80, hspace=1, fontsize=14)


# A few features seems to be (first-order) uncorrelated with the output, suggesting estimators with inherent
# feature selection should be preferred.

# Now, consider how set of base learners (estimators) perform as they are.

# In[ ]:


# We consider the following models (or base learners)
gb = XGBRegressor(n_jobs=1, random_state=SEED)
ls = Lasso(alpha=1e-6, normalize=True)
el = ElasticNet(alpha=1e-6, normalize=True)
rf = RandomForestRegressor(random_state=SEED)

base_learners = [('ls', ls),
                 ('el', el),
                 ('rf', rf),
                 ('gb', gb)
                ]


# In[ ]:


P = np.zeros((xtest.shape[0], len(base_learners)))
P = pd.DataFrame(P, columns=[e for e, _ in base_learners])

for est_name, est in base_learners:
    est.fit(xtrain, ytrain)
    p = est.predict(xtest)
    P.loc[:, est_name] = p
    print("%3s : %.4f" % (est_name, mean_absolute_error(ytest, p)))


# So they all score relatively close. However, they seem to capture different aspects of the feature space, as shown by the low correlation of their predictions:

# In[ ]:


ax = corrmat(P.corr())
show()


# They are in fact not particularly correlated in their scoring (except the linear models), and hence
# an ensemble may be able to outperform any single model by learning to combine their respective strength.

# ## 2. Comparing base learners
# 
# *emphasized text*To facilitate base learner comparison, ML-Ensemble implements a randomized grid search
# class that allows specification of several estimators (and preprocessing pipelines) in
# one grid search.

# In[ ]:


# Put their parameter dictionaries in a dictionary with the
# estimator names as keys
param_dicts = {'ls':
                  {'alpha': uniform(1e-6, 1e-5)},
               'el':
                  {'alpha': uniform(1e-6, 1e-5),
                   'l1_ratio': uniform(0, 1)},
               'gb':
                   {'learning_rate': uniform(0.02, 0.04),
                    'colsample_bytree': uniform(0.55, 0.66),
                    'min_child_weight': randint(30, 60),
                    'max_depth': randint(3, 7),
                    'subsample': uniform(0.4, 0.2),
                    'n_estimators': randint(150, 200),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'reg_lambda': uniform(1, 2),
                    'reg_alpha': uniform(1, 2),
                   },
               'rf':
                   {'max_depth': randint(2, 5),
                    'min_samples_split': randint(5, 20),
                    'min_samples_leaf': randint(10, 20),
                    'n_estimators': randint(50, 100),
                    'max_features': uniform(0.6, 0.3)}
              }


# In[ ]:


scorer = make_scorer(mean_absolute_error, greater_is_better=False)

evl = Evaluator(scorer,
                cv=2,
                random_state=SEED,
                verbose=5,
               )


# In[ ]:


evl.fit(xtrain.values,  # you can pass DataFrames from mlens>=0.1.3 
        ytrain.values,
        estimators=base_learners,
        param_dicts=param_dicts,
        preprocessing={'sc': [StandardScaler()], 'none': []},
        n_iter=2)  # bump this up to do a larger grid search


# In[ ]:


pd.DataFrame(evl.summary)


# There you have it, a comparison of tuned models in one grid search!
# 
# Optimal parameters are then easily accessed.

# In[ ]:


evl.summary["params"][('sc', 'gb')]


# # 3. Comparing meta learners
# 
# Running an entire ensemble several times just to compare different meta learners can be prohibitvely expensive. ML-Ensemble implements a class that acts as a transformer, allowing you to use ingoing layers as a "preprocessing" step, so that you need only evaluate the meta learners iteratively.

# In[ ]:


for case_name, params in evl.summary["params"].items():
    for est_name, est in base_learners:
        if est_name == case_name[1]:
            est.set_params(**params)


# In[ ]:


# We will compare a GBM and an elastic net as the meta learner
# These are cloned internally so we can go ahead and grab the fitted ones
meta_learners = [('gb', gb),
                 ('el', el)]

# Note that when we have a preprocessing pipeline,
# keys are in the (prep_name, est_name) format
param_dicts = {'el':
                  {'alpha': uniform(1e-5, 1),
                   'l1_ratio': uniform(0, 1)},
               'gb':
                   {'learning_rate': uniform(0.01, 0.2),
                    'subsample': uniform(0.5, 0.5),
                    'reg_lambda': uniform(0.1, 1),
                    'n_estimators': randint(10, 100)},
              }


# In[ ]:


# Here, we but the base learners in an EnsembleTransformer class
# this class will faithfully reproduce predictions for each fold
# in a cross-validation execution as if it was the first n layers
# of an ensemble

# The API of the Ensemble transformer mirrors that of the SequentialEnsemble class,
# see documentation for further info
in_layer = EnsembleTransformer()
in_layer.add('stack', base_learners)

preprocess = [in_layer]


# In[ ]:


evl.fit(xtrain.values,
        ytrain.values,
        meta_learners,
        param_dicts,
        preprocessing={'meta': preprocess},
        n_iter=20                            # bump this up to do a larger grid search
       )


# In[ ]:


pd.DataFrame(evl.summary)


# # 4. Ensemble learning
# 
# With these results in mind, we now turn to building an ensemble estimator.
# 
# ML-Ensemble uses a neural network-like API to specify layers of base learners to be
# fitted sequentially on the previous layer's predictions (or the raw input for the
# first layer). An ensemble is built as a Scikit-learn estimator, and can be used as
# any other Scikit-learn class.

# In[ ]:


# Let's pick the linear meta learner with the above tuned
# hyper-parameters. Note that ideally, you'd want to tune
# the ensemble as a whole, not each estimator at a time
meta_learner = meta_learners[1][1]

meta_learner.set_params(**evl.summary["params"][("meta", "el")])


# The ensemble we will implement is the Super Learner, also known as a stacking ensemble. There are several alternatives, see the documentation for further info.

# In[ ]:


# Instantiate the ensemble by adding layers to it. Finalize with a meta layer
ens = SuperLearner(verbose=5,
                   backend="threading") # mlens can release the GIL
ens.add(base_learners)
ens.add_meta(meta_learner)


# Once instantiated, the ensemble will behave like any other Scikit-learn estimator.

# In[ ]:


ens.fit(xtrain, ytrain)


# Predictions are generated as usual:

# In[ ]:


pred = ens.predict(xtest)


# In[ ]:


print("ensemble score: %.4f" % mean_absolute_error(ytest, pred))


# And that's it for this tutorial!
# 
# You might have noticed that the ensemble did not achieve an increase in performance. This is partly due to the lack of proper hyper parameter tuning, but more importantly because the base learners are not sufficiently accurate for there to be anything meaningful for the meta learner to learn from (note that predicting the average gets you about 0.07) 
# 
# In these cases, unless the meta learner is underfitting, the ensemble will at least be on par with the best base learner.  Good features are always the primary source of predictive power. Once you have them, combining different estimators in an ensemble is a powerful way of learning as much of the signal in the data as possible.
# 
# If you decide to give ML-Ensemble a try, note that the library is in beta testing so you may run into some unexpected behavior or see opportunities for improvements. Feel free to contribute to the project via the [github](https://github.com/flennerhag/mlens) repository! 
