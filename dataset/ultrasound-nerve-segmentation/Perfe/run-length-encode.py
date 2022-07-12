
import numpy as np # linear algebra

def rl_encode(img):
    x = img.transpose().flatten()
    x = x.astype('int64')
    y = np.diff(x)
    z = np.where(y>0)[0]
    w = np.where(y<0)[0]
    if x[-1] > 0:
        w = np.append(w,[len(y)-1])
    if len(z) < 10:
        return ''
    position = z + 2
    length = w - z
 
    strings = ''
    for (pos,l) in zip(position,length):
        strings = strings + str(pos) + ' ' + str(l) + ' '
    return strings
