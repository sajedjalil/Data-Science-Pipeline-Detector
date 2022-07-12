# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import datetime
import re
import csv

x=1

xlim = 20;

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
      x=x+1
      if(x<15):
        print(os.path.join(dirname, filename))
        
      if(x>15): break
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

tagword = set()
tagcount = dict()

def mutate(tw, tc, inp):
  if((inp in tw) == False):
    tc[inp] = 0;   
    tw.add(inp);
  tc[inp] = tc[inp]+1;
  pass


regx = "[^A-Za-z0-9\-]+"

def regstring(inp):
  return re.split(regx, inp);

x1 = 1;

xtime = datetime.datetime.now();

f3 = open("/kaggle/input/title-read/Paper_titles.txt")

r1 = f3.readline();

xseek = "sium"

while ((datetime.datetime.now()- xtime).seconds < xlim) and (len(r1) > 0):

  r1 = f3.readline();

  if xseek in r1.lower():
    print(r1)    

  for x in regstring(r1):
        
    mutate(tagword, tagcount, x)
    
  r1 = f3.readline();

  x1 = x1+1


f3.close()

print((datetime.datetime.now()- xtime))

print(x1," lines processed")

sort_tagcount = sorted(tagcount.items(), key=lambda x: x[1], reverse=False)

with open('bagof_wordcounts.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow(["word", "count"])

    for m in sort_tagcount:
      ##print(m)
      writer.writerow(m)

        