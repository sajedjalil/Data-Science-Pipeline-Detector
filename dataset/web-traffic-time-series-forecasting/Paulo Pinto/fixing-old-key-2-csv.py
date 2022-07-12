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

input_path = "../input/"

def quot(file):
    inp = open(input_path + file, encoding='utf-8')
    line = inp.readline().strip()
    out = open(file, "w", encoding='utf-8')
    out.write(line+'\n')
    while 1:
        line = inp.readline().strip()
        if line == '':
            break
        line = line.replace('"','""')
        line = '"'+line[:-13]+'"'+line[-13:]+'\n'
        out.write(line)
    inp.close()
    out.close()

#if __name__ == '__main__':
#    quot('key_2.csv')
#    key = pd.read_csv('key_2.csv')