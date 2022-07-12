
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns

inp_list = check_output(["ls", "../input"]).decode("utf8").split('\n')
csv_list = []
print(inp_list)
for i in inp_list:
    print(i)
    if 'csv' in i :
        csv_list.append(i)
print("cvs", csv_list)

for i in csv_list:
    sample = pd.read_csv("../input/"+i)
    print(i + ":", "total :", sample.shape)
    print(sample.head())

df = pd.read_csv('../input/train_v2.csv')
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
print(tags_counted_and_sorted.head(10))

pp = tags_counted_and_sorted.plot.bar(x='tag', y=0, figsize=(12,8))
plt.show()
plt.savefig('foo.png')