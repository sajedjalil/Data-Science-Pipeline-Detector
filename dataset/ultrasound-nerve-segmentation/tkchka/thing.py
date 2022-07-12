import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train_masks.csv')
print('Shape of training data: ' + str(df_train.shape) + '\n')
print(df_train.head())
# All our training data is in one column, titled `pixels`.
# 
# The Data page states: *train_masks.csv gives the training image masks in run-length encoded format.* So, we need to segment this data into something we can plot!

pixels = df_train['pixels'].values
print(pixels[0])
p1 = [] # Run-start pixel locations
p2 = [] # Run-lengths
p3 = [] # Number of data points per image

# Separate run-lengths and pixel locations into seperate lists
for p in pixels:
    x = str(p).split(' ')
    i = 0
    for m in x:
        if i % 2 == 0:
            p1.append(m)
        else:
            p2.append(m)
        i += 1
        
# Get number of data points in each image
# i = 0
# for p in pixels:
#     x = str(p).split(' ')
#     if len(x) == 1:
#         p3.append(0)
#     else:
#         p3.append(len(x)/2)
#     i += 1

# Get all absolute target values
targets = []
for start, length in zip(p1, p2):
    i = 0
    length = int(length)
    if start != 'nan':
        pix = int(start)
        while i <= length:
            targets.append(pix)
            pix += 1
            i += 1

# Remove NaNs
p4 = []
i = 0
for p in p1:
    if p == 'nan':
        i += 1
    else:
        p4.append(p)
p1 = p4


from collections import Counter
data = Counter(targets)
l = [x[0] for x in data.most_common(500)]

print(l)
l.sort()
print(l)

d = ''
for x in l:
    d = d + str(x) + ' 1 '

print(d)

sub = pd.read_csv('../input/sample_submission.csv')
print(sub)

sub['pixels'] = d

print(sub)

sub.to_csv('com25.csv', index=False)