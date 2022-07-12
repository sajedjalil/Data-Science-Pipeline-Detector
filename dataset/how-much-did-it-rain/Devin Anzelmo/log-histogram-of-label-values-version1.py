# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from pylab import savefig

pd.options.mode.chained_assignment = None
#counts the number of radar scans in the row by using white spaces
def get_num_radar_scans(timetoend_row):
    return timetoend_row.count(' ') + 1

#returns a subset of the data with with number of radar scans between upper and lower bound
#additionally appends a partition name column to be used later for plotting. 
def partition_on_count(data_df,lower_bound, upper_bound):
    if lower_bound == upper_bound:   
        part =  data_df.query('num_radar_scans == @lower_bound')
        part['Number of Scans']  = str(lower_bound) +  '_scan'
        return part, str(lower_bound) +  '_scan'
    else:
        part =  data_df.query('num_radar_scans > @lower_bound and num_radar_scans < @upper_bound')
        part['Number of Scans']  = str(lower_bound+1) + '-' + str(upper_bound-1)
        return part, str(lower_bound+1) + '-' + str(upper_bound-1)

z = zipfile.ZipFile('../input/train_2013.csv.zip')
train = pd.read_csv(z.open('train_2013.csv'),usecols=['Expected','TimeToEnd'])

#calculate the number of scans in each row.
number_scans = train.TimeToEnd.apply(get_num_radar_scans)

#drop time to end as it is no longer needed
train.drop('TimeToEnd',axis=1,inplace=True)

train = pd.concat([train, number_scans],axis=1)
train.columns = ['Expected','num_radar_scans']

bounds =[[0,8],[7,18],[17,100]]

partitioned_train = pd.DataFrame()
partition_names = []
#this adds a third column to the data from which allows for easy plotting with seaborn 
for i in bounds:
    partition,partition_name = partition_on_count(train,i[0],i[1])
    partitioned_train = pd.concat([partitioned_train,partition ])
    partition_names.append(partition_name)

#round all the labels up, this is required to use a classifier on the problem
partitioned_train.loc[:,'Expected'] = partitioned_train.Expected.apply(np.ceil)

#get rid of large rain gauge readings as they are mainly noise, and hurt the score
partitioned_train = partitioned_train.query('Expected < 70')

sns.set(style="darkgrid")
g = sns.FacetGrid(partitioned_train, col="Number of Scans", margin_titles=False,col_order=partition_names,size=3.6)
g.map(plt.hist, "Expected", color="steelblue", bins=70, lw=.85,log=True,normed=True)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Histograms of Label values', fontsize=15)
g.set(yticklabels=[])
g.set_axis_labels( 'Expected(mm)','Log Proportion')
g.savefig('figure2.png')

