import pandas as pd
import random
import csv

f = "../input/train_ver2.csv"
num_lines = sum(1 for l in open(f))
size = int(num_lines / 1000)
skip_idx = random.sample(range(1, num_lines), num_lines - size)
data = pd.read_csv(f, skiprows=skip_idx, header = None )

fp = open('test.csv', 'w', newline='') 
a = csv.writer(fp, delimiter=',')
data.to_csv(fp)
fp.close


