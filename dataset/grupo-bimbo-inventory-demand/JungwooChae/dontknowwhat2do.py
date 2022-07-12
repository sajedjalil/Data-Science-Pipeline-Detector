# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

from collections import defaultdict


def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    total = 0

    client_product_arr = defaultdict(int)
    client_product_arr_count = defaultdict(int)

    # Calc counts
    avg_target = 0.0
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        week = int(arr[0])
        agency = arr[1]
        canal_id = arr[2]
        ruta_sak = arr[3]
        cliente_id = int(arr[4])
        producto_id = int(arr[5])
        vuh = arr[6]
        vh = arr[7]
        dup = arr[8]
        dp = arr[9]
        target = int(arr[10])
        
run_solution()

# Any results you write to the current directory are saved as output.

