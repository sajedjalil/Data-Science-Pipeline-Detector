# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
test = pd.read_json("../input/test.json")
train = pd.read_json("../input/train.json")
sample_submission = pd.read_csv("../input/sample_submission.csv")
train.head()
# add column to count number of ingredients per recipe
train['number_of_ingredients'] = train.ingredients.str.len()
# group by cuisine
groupby_cuisine = train.groupby(train['cuisine'])
# calculate some stats per cuisine
groupby_stats = groupby_cuisine['number_of_ingredients'].agg({'number_of_recipes' : np.size,'avg_number_ingredients' : np.mean})
# plot the summary stats
fig = plt.figure()
ax = groupby_stats['number_of_recipes'].plot(kind="bar",color='red')
ax.set_ylabel('Number of Recipes')
ax.yaxis.label.set_color('red')
ax.tick_params(axis='y', colors='red')
plt.xticks(rotation='vertical')
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(),groupby_stats['avg_number_ingredients'],marker='o',color='blue')
ax2.set_ylabel('Avg Number of Ingredients')
ax2.yaxis.label.set_color('blue')
ax2.tick_params(axis='y', colors='blue')
plt.title('Stats by Cuisine');
