""" 
A short code snippet to run tpot, an auto-ml tool using genetic programming
https://en.wikipedia.org/wiki/Genetic_programming.
The github repo is here for more details: https://github.com/EpistasisLab/tpot
"""

from tpot import TPOTClassifier
import pandas as pd


SEED = 314 # PI day, anyone?
# Small values so that it runs
CV = 5
GENERATIONS = 10
POPULATION_SIZE = 10
N_SAMPLES = 1000

def tpot_pipeline():
    model = TPOTClassifier(scoring='roc_auc', early_stop=1000, n_jobs=-1, 
                           random_state=SEED, cv=CV, verbosity=2, generations=GENERATIONS,
                           population_size=POPULATION_SIZE)
    
    
    train_df = pd.read_csv("../input/train.csv").drop("ID_code", axis=1).sample(N_SAMPLES)
    TARGET_COL = 'target'
    FEATURES_COLS = train_df.drop(TARGET_COL, axis=1).columns.tolist()
    
    
    
    model.fit(train_df[FEATURES_COLS].values, train_df[TARGET_COL].values)
    model.export('tpot_pipeline.py')
    
if __name__ == "__main__":
    tpot_pipeline()