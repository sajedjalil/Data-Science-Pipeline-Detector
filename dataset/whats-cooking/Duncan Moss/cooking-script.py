import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Plot Style
plt.style.use(u'ggplot')

# Read in training Data File
# Fields are cuisine, id, ingredients
train = pd.read_json("../input/train.json")

# Find Cuisine Distribution
cuisine_distribution = Counter(train.cuisine)

# Plot Cuisine Distribution
cuisine_fig = pd.DataFrame(cuisine_distribution, index=[0]).transpose()[0].sort(ascending=False, inplace=False).plot(kind='barh')
cuisine_fig.invert_yaxis()
cuisine_fig = cuisine_fig.get_figure()
cuisine_fig.tight_layout()
cuisine_fig.savefig("Cuisine_Distribution.jpg")

# Find Ingredient Distribution
recipe_ingredient = [Counter(recipe) for recipe in train.ingredients]
ingredient_distribution = sum(recipe_ingredient, Counter())

# Plot Ingredient Distribution
ingredient_fig = pd.DataFrame(ingredient_distribution, index=[0]).transpose()[0].sort(ascending=False, inplace=False)[:20].plot(kind='barh')
ingredient_fig.invert_yaxis()
ingredient_fig = ingredient_fig.get_figure()
ingredient_fig.tight_layout()
ingredient_fig.savefig("Ingredient_Distribution.jpg")
