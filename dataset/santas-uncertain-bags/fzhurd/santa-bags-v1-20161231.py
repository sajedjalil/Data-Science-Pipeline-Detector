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

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.random.seed(1000)

""" 
    This script refer from https://www.kaggle.com/mchirico/santas-uncertain-bags/santa-quick-look
    and from https://www.kaggle.com/sudalairajkumar/santas-uncertain-bags/santa-btb-with-validation/code
"""

def Weight(mType):

    if mType == "horse":
        return max(0, np.random.normal(5,2,1)[0])
    if mType == "ball":
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    if mType == "bike":
        return max(0, np.random.normal(20,10,1)[0])
    if mType == "train":
        return max(0, np.random.normal(10,5,1)[0])
    if mType == "coal":
        return 47 * np.random.beta(0.5,0.5,1)[0]
    if mType == "book":
        return np.random.chisquare(2,1)[0]
    if mType == "doll":
        return np.random.gamma(5,1,1)[0]
    if mType == "blocks":
        return np.random.triangular(5,10,20,1)[0]
    if mType == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]


def fill_bags_m1():
    total_weight = 0
    missed_count = 0
    for i in range(1000):
            block_w = Weight('blocks')
            ball_w = Weight('ball')
            train_w = Weight('train')
            bike_w = Weight('bike')
            doll_w = Weight('doll')
            horse_w = Weight('horse')
            coal_w = Weight('coal')
            book_w = Weight('book')
            gloves_w = Weight('gloves')
            sum_w = train_w + block_w + horse_w + doll_w + book_w + ball_w
            if sum_w < 50.:
                total_weight += sum_w
            else:
                missed_count += 1

    print("final total weights : ", total_weight)
    print("Bags that are not passed : ",missed_count)

    save_results("results.csv", 'w')


def save_results(file, mode):
    with open(file, mode) as f:
        f.write("Gifts\n")
        for i in range(1000):
            f.write('train_'+str(i)+' blocks_'+str(i)+' horse_'+str(i)+' doll_'+str(i)+' book_'+str(i)+' ball_'+str(i)+'\n')

def main():
    fill_bags_m1()

if __name__=='__main__':
    main()