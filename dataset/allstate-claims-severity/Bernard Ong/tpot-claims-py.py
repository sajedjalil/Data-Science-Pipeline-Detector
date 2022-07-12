# ------------------------------------------------------------------------------
# Created by: Bernard Ong (bernard.ong@entense.com)
# Created on: Nov 16, 2016
# Created to: Educational purposes only and idea generation :)
# Purpose to: Experiment with a Genetic Algorithm for Model Selection and
#             Hyperparamter Optimization
# Applied to: AllState Claims Severity Training Dataset
# ------------------------------------------------------------------------------

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

t0 = time.time()

print ("-" * 100)
print ("Load Training Data Only")
train = pd.read_csv('../input/train.csv')
print ("Total %i Rows Found" % (train.shape[0]))

print ("-" * 100)
print ("Factorize Categorical Data")
features = train.columns
cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train[feat] = pd.factorize(train[feat], sort=True)[0]

print ("-" * 100)
print ("Load Predictors and Labels")
X = train.drop(['id','loss'], axis=1).values
y = train.loc[:,'loss'].values

print ("-" * 100)
print ("Create the Training and Validation Folds")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.70,
    test_size=0.30
    )

print ("-" * 100)
print ("Instantiate the TPOT Regressor Model")
my_tpot = TPOTRegressor(
    generations = 5,                     # the more the merrier but longer
    #population_size = 100,
    #crossover_rate = 0.1,               # 0.0-1.0
    #mutation_rate = 0.1,                # 0.0-1.0
    #num_cv_folds = 10,                  # 2-10
    scoring = 'mean_absolute_error',
    #max_time_mins = 5,
    #max_eval_time_mins = 60,
    random_state = 0,
    verbosity = 3                        # 0,1,2,3
    )

print ("-" * 100)
print ("Train and Fit TPOT Model")
my_tpot.fit(X_train, y_train)

print ("-" * 100)
print ("Mean Absolute Error = %.6f" % (my_tpot.score(X_test, y_test)))

my_tpot.export('tpot_gen_code.py')

t1 = time.time()
print ("-" * 100)
print ("Completed Genetic Modeling in %.2f minutes" % ((t1-t0)/60))
print ("\a")

