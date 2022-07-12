import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

from deap import base
from deap import creator
from deap import tools

#######################
# Load & process data
#######################
def get_data():
    #################
    # read datasets
    #################
    train = pd.read_csv('../input/train.csv')
    test_submit = pd.read_csv('../input/test.csv')

    # Get y and ID
    train = train[train.y < 250] # Optional: Drop y outliers
    y_train = train['y']
    train = train.drop('y', 1)
    test_submit_id = test_submit['ID']

    #########################
    # Create data
    #########################
    features = ['X0',
                'X5',
                'X118',
                'X127',
                'X47',
                'X315',
                'X311',
                'X179',
                'X314',
                'X232',
                'X29',
                'X263',
                'X261']

    # Build a new dataset using key parameters, lots of drops
    train = train[features]
    test_submit = test_submit[features]

    # Label encoder
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test_submit[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test_submit[c] = lbl.transform(list(test_submit[c].values))

    # Convert to matrix
    train = train.as_matrix()
    y_train = np.transpose([y_train.as_matrix()])
    test_submit = test_submit.as_matrix()
    test_submit_id = test_submit_id.as_matrix()

    return train, y_train, test_submit, test_submit_id

#########################
# XGBoost Model
#########################
def gradient_boost(train, y_train, params):
    y_mean_train = np.mean(y_train)

    # prepare dict of params for xgboost to run with
    xgb_params = {
        'n_trees': params[0],
        'eta': params[1],
        'max_depth': params[2],
        'subsample': params[3],
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean_train,
        'seed': 123456789,
        'silent': 1
    }

    # form DMatrices for Xgboost training
    dtrain = xgb.DMatrix(train, y_train)

    # xgboost, cross-validation
    cv_result = xgb.cv(xgb_params,
                       dtrain,
                       nfold = 10,
                       num_boost_round=5000,
                       early_stopping_rounds=100,
                       verbose_eval=False,
                       show_stdv=False
                      )

    num_boost_rounds = len(cv_result)

    # train model
    model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)

    # get model accuracy
    accuracy = r2_score(dtrain.get_label(), model.predict(dtrain))

    return model, accuracy

######################
# Genetic algorithm
######################
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Get data
train, y_train, test_submit, test_submit_id = get_data()

# Attribute generator
toolbox.register("n_trees", random.randint, 100, 10000)
toolbox.register("eta", random.uniform, 0.0001, 0.01)
toolbox.register("max_depth", random.randint, 1, 10)
toolbox.register("subsample", random.uniform, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_trees, toolbox.eta, toolbox.max_depth, toolbox.subsample), n=1)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    model, accuracy = gradient_boost(train, y_train, individual)
    return [accuracy]

# the model for a specified individual
def getModel(individual):
    model, accuracy = gradient_boost(train, y_train, individual)
    return model

# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(12345)

    # create an initial population
    pop = toolbox.population(n=20)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual so far is %s, %s" % (best_ind, best_ind.fitness.values))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()