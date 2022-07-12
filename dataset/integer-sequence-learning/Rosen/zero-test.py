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

#coding = utf-8

import csv

TEST_DATA = '../input/test.csv'
TEST_RESULT = 'test_result.csv'

def test():
    test_file = open(TEST_DATA, 'r')
    result_file = open(TEST_RESULT, 'w')
    reader = csv.reader(test_file)
    writer = csv.writer(result_file)
    writer.writerow([r'Id',r'Last'])
    for line in reader:
        if reader.line_num == 1: continue
        writer.writerow([line[0], 0])
    test_file.close()
    result_file.close()

if __name__ == '__main__':
    test()