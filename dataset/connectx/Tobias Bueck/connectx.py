# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import kaggle_environments

# %% [code]
env = kaggle_environments.make("connectx", debug=True)
env.render()

# %% [code]
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

# %% [code]
env.reset()
# Play as the first agent against default "random" agent.
env.run([None, "random"])
env.render(mode="ipython", width=500, height=450)

# %% [code]
ROW = 7
COLUMN = 6

# %% [code]
from random import randint

# %% [code]
example_field = list([randint(0,2) for i in range(4*4)])

# %% [code]
print(example_field)

# %% [code]


# %% [code]
def get_columns(list_field, r_l, c_l):
    columns = []
    for i in range(r_l):
        new_column = []
        for k in range(i, r_l*c_l, r_l):
            new_column.append(list_field[k])
        columns.append(new_column)
    return columns    

# %% [code]
lol = 1 if False else 2

# %% [code]
lol

# %% [code]
def get_all_vertical(game_field, length):
    all_columns = []
    for column in game_field:
        for i in range(length, len(column)+1):
            values = column[i-length:i]
            value_before = column [i-length-1] if i - length - 1 > 0 else -1
            value_after = column[i + 1]  if i + 1 < len(column) else -1
            column_entry = (value_before, values, value_after)
            all_columns.append(column_entry)
    return all_columns

# %% [code]
def test_correct(window, agents_disc, zero_test=True):
    if zero_test:
        return all(map(lambda x: x == agents_disc, window[1])) and (window[0] == 0 or window[2] == 0)
    else:
        return all(map(lambda x: x == agents_disc, window[1]))

# %% [code]
def correct_vertical(game_field, agents_disc, length, zero_test=True):
    all_verts = get_all_vertical(game_field, length)
    return list(map(lambda vert: test_correct(vert, agents_disc, zero_test), all_verts)).count(True)

# %% [code]
def get_rows_from_columns(game_field):
    column_length = len(game_field[0])
    rows = []
    for i in range(column_length):
        row = [game_field[k][i] for k in range(len(game_field))]
        rows.append(row)
    return rows

# %% [code]
"""def get_all_horizontal(game_field, length):
    rows = get_rows_from_columns(game_field)
    return get_all_vertical(rows, length)
"""

# %% [code]
def correct_horizontal(game_field, agents_disc, length, zero_test=True):
    rows = get_rows_from_columns(game_field)
    return correct_vertical(rows, agents_disc, length, zero_test)

# %% [code]
def put_disc_in(game_field, disc, column):
    new_game_field = game_field[:]
    new_game_field[column]
    last_zero = -1
    for i in range(len(new_game_field[column])):
        if new_game_field[column][i] != 0:
            last_zero = i
    new_game_field[column][last_zero] = disc
    return new_game_field

# %% [code]
from random import choice
import random

# %% [code]
def get_score(game_field, disc, enemy_disk, inarow):
    score = 0
    if inarow > 0:
        score += 1_000_000_000 * (correct_vertical(game_field, disc, inarow, False) + correct_horizontal(game_field, disc, inarow, False))
    if inarow - 1 > 0:
        score -= 1_000 * (correct_vertical(game_field, enemy_disk, inarow - 1) + correct_horizontal(game_field, enemy_disk, inarow - 1))
        score += 100 * (correct_vertical(game_field, disc, inarow - 1) + correct_horizontal(game_field, disc, inarow - 1))
    if inarow - 2 > 0:
        score -= 100 * (correct_vertical(game_field, enemy_disk, inarow - 2) + correct_horizontal(game_field, enemy_disk, inarow - 2))
        # score += 100 * (correct_vertical(game_field, disc, inarow - 2) + correct_horizontal(game_field, disc, inarow - 2))
    return score

# %% [code]
test = {"tobi": 10, "laura": 5}
max(list(test.items()), key=lambda x: x[1])
#list(test.items())

# %% [code]
def make_move(game_field, options, disc, inarow):
    enemy_disc = [1, 2]
    enemy_disc.remove(disc)
    enemy_disc = enemy_disc[0]
    score_options = {option: 0 for option in options}
    for opt in options:
        new_game_field = put_disc_in(game_field,disc, opt)
        score = get_score(new_game_field, disc, enemy_disc, inarow)
        score_options[opt] = score
    return max(list(score_options.items()), key=lambda x: x[1])[0]

# %% [code]
def get_all_possible_columns(game_field):
    possible = []
    for counter, column in enumerate(game_field):
        if column[0] == 0:
            possible.append(counter)
    return possible

# %% [code]
def my_second_agent(observation, configuration):
    row_length = configuration.columns
    column_length = configuration.rows
    game_field = get_columns(observation.board, row_length, column_length)
    options = get_all_possible_columns(game_field)
    disc = observation.mark
    return make_move(game_field, options, disc, configuration.inarow)
    

# %% [code]
cc = [[0,1,0,0], [1, 0, 0, 1], [0, 2, 1, 2], [2, 1, 0, 1]]

# %% [code]
get_rows_from_columns(cc)

# %% [code]
get_all_vertical(cc, 2)

# %% [code]
env.reset()
# Play as the first agent against default "random" agent.
env.run([my_second_agent, "random"])
env.render(mode="ipython", width=500, height=450)

# %% [code]
from kaggle_environments import evaluate

# %% [code]
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

# %% [code]
get_win_percentages(my_second_agent, "random", 500)

# %% [markdown]
# 1. created an agent better than random !! yeah!!**

# %% [code]
