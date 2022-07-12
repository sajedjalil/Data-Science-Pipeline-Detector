import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np

def RLenc(bytes ,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    # bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs


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
l = [x[0] for x in data.most_common(14000)]

#print(l)
l.sort()
#print(l)

pixels = []
for _ in range(0, 243600):
    pixels.append(0)
    
for x in l:
    p = x - 1
    pixels[p] = 1
    
runs = RLenc(pixels)

# d = ''
# for x in l:
#     d = d + str(x) + ' 1 '

print(runs)

sub = pd.read_csv('../input/sample_submission.csv')
print(sub)

sub['pixels'] = runs

print(sub)

sub.to_csv('com14k.csv', index=False)