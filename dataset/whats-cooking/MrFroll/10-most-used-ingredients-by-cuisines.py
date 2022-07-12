#
# This script transforms the 'ingredients' feature of the original dataset
# into separated columns. The values of these new columns are True or False,
# depending if the ingredient was used on the receipe.
#
# The ingredients need to be cleaned, making 'low fat mozzarella' and 
# 'reduced fat mozzarella' the same ingredient. Ideas are welcome.
# 

import pandas as pd
import matplotlib.pyplot as plt

# reading the file as dataframe and getting summary #
df = pd.read_json('../input/train.json')

# grouping the rows based on id and get the rainfall #
df_grouped = df.groupby(['cuisine'])

for cuisine in df_grouped.groups.keys():
    print('{0:15} -> {1:10}'.format(cuisine,len(df_grouped.groups[cuisine])))

plt.style.use(u'ggplot')
all_ingredients = set()
df.ingredients.map(lambda x: [all_ingredients.add(i) for i in list(x)])

# fill the dataset with a column per ingredient
for ingredient in all_ingredients:
    df[ingredient] = df.ingredients.apply(lambda x: ingredient in x)


# Lets take a serie with the number of times each ingredient was used
for cuisine in df_grouped.groups.keys():
    df_cuisine = df_grouped.get_group(cuisine);
    s = df_cuisine[list(all_ingredients)].apply(pd.value_counts).fillna(0).transpose()[True]
    # Finally, plot the 10 most used ingredients
    fig = s.sort(inplace=False, ascending=False)[:10].plot(kind='bar', title = cuisine)
    fig = fig.get_figure()
    fig.tight_layout()
    fig.savefig(cuisine + '_10_most_used_ingredients.jpg')