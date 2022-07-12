# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import csv

# Any results you write to the current directory are saved as output.

p_train = '../input/train.csv'
p_test = '../input/test.csv'
p_out = 'pred.csv'

c_stat = {}
nrows = 0
for row in csv.DictReader(open(p_train)):
    did = row['srch_destination_id']
    cid = row['hotel_cluster']
    book = row['is_booking']
    c_stat[cid] = c_stat.get(cid, 0) + int(book)
    nrows += 1
    #if nrows > 1000: break

print(c_stat)
c_hot = ' '.join([k for k,v in sorted(c_stat.items(),key=lambda d:-d[1])[:5]])
print(c_hot)

nrows = 0
with open(p_out, 'w') as fo:
    fo.write('id,hotel_cluster\n')
    for row in csv.DictReader(open(p_test)):
        eid = row['id']
        did = row['srch_destination_id']
        fo.write('%s,%s\n' % (eid, c_hot))
        nrows += 1
        #if nrows > 100:    break
    
    