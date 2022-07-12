# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sample = pd.read_csv('../input/train.csv', usecols=['hotel_cluster'])

sample.head()



destination = pd.read_csv('../input/destinations.csv')
destination.head()


children=pd.read_csv('../input/train.csv', usecols=['srch_children_cnt'])
counts = pd.value_counts(children.ix[:,0]) 
counts     
    

booking=pd.read_csv('../input/train.csv', usecols=['is_booking'])
counts = pd.value_counts(booking.ix[:,0]) 
counts  

adult=pd.read_csv('../input/train.csv', usecols=['srch_adults_cnt'])
counts = pd.value_counts(adult.ix[:,0]) 
counts   
hotel_cl=pd.read_csv('../input/train.csv', usecols=['hotel_cluster'])
counts = pd.value_counts(hotel_cl.ix[:,0]) 
counts  
