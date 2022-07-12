import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

pal = sns.color_palette()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

print('# File sizes')
for f in os.listdir('../input'):
    if not os.path.isdir('../input/' + f):
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
    else:
        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]
        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))

df = pd.read_csv("../input/train_v2.csv")
tags = df["tags"].apply(lambda x: x.split(' '))
   
end = len(tags)
id_haze = []
id_cloudy = []
id_partly = []
id_clear = []

for i in range (0,end):
    for x in tags[i]:
        if x == 'haze':
            id_haze.append(i)
        elif x == 'cloudy':
            id_cloudy.append(i)
        elif x == 'partly-cloudy':
            id_partly.append(i)
        elif x == 'clear':
            id_clear.append(i)
print (len(id_haze))
print (len(id_cloudy))
print (len(id_partly))
print(len(id_clear))