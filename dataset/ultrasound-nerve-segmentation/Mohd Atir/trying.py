import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# This function takes a list of pixels, and returns them in run-length encoded format.
def PixelsToRLenc(pixels ,order='F',format=True):
    """
    Based off code by https://www.kaggle.com/alexlzzz
    pixels is a list of absolute pixel values which need to be converted. (1-243600)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    
    # Initialse empty array
    bytes = []
    for _ in range(0, 243600):
      bytes.append(0)
    
    # Place values from input list into the array
    for x in pixels:
        p = x - 1
        bytes[p] = 1
    
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
    
pixels = df_train['pixels'].values
print(pixels[0])

# This function takes a string in run-length encoded format, and returns a list of pixels
def RLencToPixels(runs):
p1 = [] # Run-start pixel locations
p2 = [] # Run-lengths
    
# Separate run-lengths and pixel locations into seperate lists
x = str(runs).split(' ')
i = 0
for m in x:
    if i % 2 == 0:
        p1.append(m)
    else:
        p2.append(m)
    i += 1
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

print(l)
l.sort()
print(l)
    
# Get all absolute pixel values
pixels = []
for start, length in zip(p1, p2):
    i = 0
    length = int(length)
    pix = int(start)
    while i < length:
        pixels.append(pix)
        pix += 1
        i += 1
            
return pixels
    
runs = RLenc(pixels)
print(runs)
sub = pd.read_csv('../input/sample_submission.csv')
    print(sub)

    sub['pixels'] = runs

    print(sub)

    sub.to_csv('com14k.csv', index=False)