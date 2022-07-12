
# I was keeping adding features and found out that some features actually hurt the
# performance of the model. Greedily removing the least important features does
# not help too much. I borrowed code from
# https://github.com/scoliann/GeneticAlgorithmFeatureSelection/blob/master/gaFeatureSelectionExample.py,
# which uses genetic algorithm to select best features. After running the script,
# my local cv score for 10 million rows was improved from 0.9783 to 0.9794.


import numpy as np
import pandas as pd
import gc
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from deap import creator, base, tools, algorithms
import random

params = {
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'goss',

        'min_data_in_leaf': 5000,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'max_bin': 255,

        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200,
}

NUM_TRAIN = 100000 # 10000000
NUM_CV = 50000 # 5000000

data = pd.read_csv("../input/train.csv", skiprows=range(1, 184903891-NUM_TRAIN-NUM_CV))

X_train = data[:NUM_TRAIN]
X_cv = data[NUM_TRAIN:]
del data
gc.collect()

Y_train = X_train["is_attributed"]
X_train = X_train.drop(["is_attributed"], axis=1)
Y_cv = X_cv["is_attributed"]
X_cv = X_cv.drop(["is_attributed"], axis=1)

def getFitness(individual, X_train, X_test, y_train, y_test):

        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
        X_testParsed = X_test.drop(X_test.columns[cols], axis=1)

        gbm_train = lgbm.Dataset(X_train, y_train)
        gbm_cv = lgbm.Dataset(X_test, y_test)

        clf = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
        cv_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
        score = roc_auc_score(y_test, cv_pred)

        print("Individual: {}  Fitness_score: {} ".format(individual,score))
        return (score,)

#========DEAP GLOBAL VARIABLES ========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Create Operators
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_cv, y_train=Y_train, y_test=Y_cv)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create population, hall of fame
numPop = 20  # Number of population
numGen = 10  # Number of generation
pop = toolbox.population(n=numPop)
hof = tools.HallOfFame(numPop)

# Add statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Launch genetic algorithm
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

# Best candidates are stored in the hall of fame
for individual in hof:
        print(individual)
