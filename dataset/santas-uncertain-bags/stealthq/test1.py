# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

with open('../input/gifts.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if reader.line_num == 1: 
            continue
        p = re.compile('^(\w+)_(\d+)$')
        m = p.match(row.pop())
        giftId = m.group(0)
        giftType = m.group(1)
        giftNum = m.group(2)
        print('{} {} {}'.format(giftId, giftType, giftNum))
    