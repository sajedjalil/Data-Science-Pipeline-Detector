import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

distinct = train.groupby('place_id').count().sort_values('row_id',ascending=False)
distinct.head()
single_location=train[train['place_id']==8772469670]
#single_location['x']=single_location['x'].round(2)
#single_location['y']=single_location['y'].round(2)
single_location=single_location[['place_id','x','y','accuracy']].groupby(['place_id','x','y'])['accuracy'].mean()
single_location.head()
