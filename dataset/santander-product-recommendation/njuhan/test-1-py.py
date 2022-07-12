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


def load_file_byline(in_file, out_file, size =  10):
    line_count = 0
    out_f = open(out_file,"w+")
    in_f = open(in_file, 'r')
    
    line = in_f.readline()
    while line:
        line_count += 1
        if line_count>size:
            break
        
        out_f.write(line)
        line = in_f.readline()
        
    
    out_f.close()
    in_f.close()
    print('load down')
    
        

def load_data(in_file, out_file):
    in_f = open(in_file,'r')
    out_f = open(out_file, 'w')
    
    lines = in_f.readlines()
    first_line = lines[0]
    rest_lines = lines[1:]
    print('data size:', len(rest_lines))
    
    out_f.writelines(rest_lines[-10:])
    
    in_f.close()
    out_f.close()
    
    
    
    
    
    

if __name__ == '__main__': 
    
    in_file = '../input/train_ver2.csv'
    out_file = 'out.txt'
    #load_file_byline(in_file, out_file, 1000)
    
    load_data(in_file, out_file)
    
    