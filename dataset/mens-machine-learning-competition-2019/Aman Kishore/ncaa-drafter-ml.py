# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import csv
import random
from sklearn import linear_model, model_selection
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/datafiles"))

df = pd.read_csv("../input/datafiles/RegularSeasonCompactResults.csv")
print(df)

# Load the data
folder = '../input'
season_data = pd.read_csv(folder + '/datafiles/RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv(folder + '/datafiles/NCAATourneyDetailedResults.csv')
seeds = pd.read_csv(folder + '/datafiles/NCAATourneySeeds.csv')
frames = [season_data, tourney_data]
all_data = pd.concat(frames)
stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr',
                   'ast', 'to', 'stl', 'blk', 'pf']
prediction_year = 2018
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
submission_data = []
def initialize_data():
    for i in range(1985, prediction_year+1):
        team_elos[i] = {}
        team_stats[i] = {}
initialize_data()

all_data.head(10) # Gets the top 10 data

#hello aman