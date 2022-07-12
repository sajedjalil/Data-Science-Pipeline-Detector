# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import re
import time
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# defining pattern to identify time and date from click_time
pattern = r'\s*\d+-\d+-(\d+)\s+(\d+):\d+:\d+'
pattern_regex = re.compile(pattern)

# connecting and creating the SQLite database
conn = sqlite3.connect('talkingdata.db')
c = conn.cursor()
# creating table and defining column. I have divided click_time into click_time_hr and click_time_day
c.execute('''CREATE TABLE IF NOT EXISTS train_full (
            ip INTEGER,
            app INTEGER,
            device INTEGER,
            os INTEGER,
            channel INTEGER,
            click_time_day INTEGER,
            click_time_hr INTEGER,
            is_attributed INTEGER)''')

# reading the file line by line
i = 0
start = time.time()
for line in open("../input/train_sample.csv"):  # I am working on train_sample file here but you can change it to train.csv on your system and work on it 
    if i == 0:
        i += 1
        continue
    csv_row = line.split(',')
    #finding date and hr from click_time
    result = pattern_regex.findall(csv_row[5])
    i += 1
    print("Inserting line no :" + str(i))
    c.execute("INSERT INTO train_full(ip,app,device,os,channel,click_time_day,click_time_hr,is_attributed) VALUES(?,?,?,?,?,?,?,?)",\
              (csv_row[0],csv_row[1],csv_row[2],csv_row[3],csv_row[4],result[0][0],result[0][1],csv_row[7]))

conn.commit()
c.close()
conn.close()

end = time.time()
print("total time taken ")
print(end - start)


