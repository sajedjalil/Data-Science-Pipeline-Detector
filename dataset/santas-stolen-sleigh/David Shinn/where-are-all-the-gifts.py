import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

gifts = pd.read_csv('../input/gifts.csv')
gifts.info()
gifts.head()
gifts.describe()

# All gift locations
gifts.plot.scatter('Longitude', 'Latitude', alpha=0.3, s=1, color='brown')
plt.title('Gift Locations; N = {:,}'.format(len(gifts)))
plt.show()
plt.savefig('All_gift_locations.png')

# Create category for Weight = 1 and quintile of other weights
gifts['Weight_Group'] = pd.qcut(gifts.Weight.replace(1, np.nan), 5)
gifts.Weight_Group = gifts.Weight_Group.cat.add_categories('[1, 1]').fillna('[1, 1]')
categories = list(gifts.Weight_Group.cat.categories)
categories = [categories[-1]] + categories[:-1]
gifts.Weight_Group = gifts.Weight_Group.cat.reorder_categories(categories)

gifts.groupby('Weight_Group').size()

fig, axes = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(top=0.90)
plt.suptitle('Location of Minimum Weight Gifts and Quintiles of Other Weights Gifts')
fig.set_size_inches(15, 7)
for count, (group, df) in enumerate(gifts.groupby('Weight_Group')):
    row = math.floor(count / 3)
    col = count % 3
    df.plot.scatter('Longitude', 'Latitude', alpha=0.5, s=1, ax=axes[row][col], color='brown')
    axes[row][col].set_title('Weights {:}; N = {:,}'.format(group, len(df)))
plt.subplots_adjust(hspace=0.4)
plt.show()
plt.savefig('Gift_locations_by_weight.png')