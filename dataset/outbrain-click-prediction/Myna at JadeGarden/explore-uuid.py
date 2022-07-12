# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections #I had a list of lists going, the good people at Stack Overflow said DefaultDict!
import matplotlib.pyplot as plt
pd.set_option('max_rows',10)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
events = pd.read_csv("../input/events.csv",chunksize=100000)
total = 0
for chunk in events:
    cc = chunk['uuid'].value_counts()
    try:
        bb = pd.concat([bb,cc])
    except:
        bb = cc

dd = bb.groupby(bb.index).sum()

occurence = dd.value_counts()
print(occurence)

plt.bar(occurence.index,occurence,log=True)
plt.xlabel('n = # of times a uuid appearing in events.csv')
plt.ylabel('# of uuid that appear for n times')
plt.savefig('Explore_uuid.png')
plt.show()