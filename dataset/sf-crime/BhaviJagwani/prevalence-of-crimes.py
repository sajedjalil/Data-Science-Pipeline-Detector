import pandas as pd
import numpy as np 
import string

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('ggplot')

df = pd.read_csv('../input/train.csv', header=0)

df= df.drop(['Descript','X','Y'], axis=1)

tmp= pd.DataFrame(df.Dates.str.split().tolist(), columns="date time".split())

df['Hour']= tmp.apply(lambda row: row['time'].split(':')[0], axis=1)

df= df.drop(['Dates'], axis=1)

df.Cat= df.Category.astype('category')

cat_dict= dict(zip(df.Cat.cat.categories, range(len(df.Cat.cat.codes))))

df.dist= df.PdDistrict.astype('category')
dist_dict= dict(zip(df.dist.cat.categories, range(len(df.dist.cat.codes))))

cat_dist= pd.pivot_table(df, index="Category", columns="PdDistrict", values="DayOfWeek", aggfunc="count")

cplot= cat_dist.plot(kind='barh', figsize=(30,15), grid=0, colormap=cm.rainbow, stacked= True)
plt.savefig("stackedBar3",ext="png", transparent=0)
