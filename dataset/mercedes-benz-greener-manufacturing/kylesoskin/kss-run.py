# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

reader = csv.DictReader(open('../input/train.csv'))

result = {}
for row in reader:
    key = row.pop('ID')
    if key in result:
        # implement your duplicate row handling here
        pass
    result[key] = row
    
for car_id in result.items():
    print(car_id)






