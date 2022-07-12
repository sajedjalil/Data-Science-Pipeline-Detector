import math
import numpy as np
import pandas as pd

im_size = [1918, 1280]
h_skip = 370
w_skip_min = 150
w_skip_max = 600

ss = pd.read_csv('../input/sample_submission.csv')
d = {}
def get_str(i):
    w_skip = w_skip_min + round(abs(math.cos(2*math.pi*(i-1)/16))*(w_skip_max - w_skip_min))

    str_buf = ''
    for c in range(im_size[1] - h_skip*2):
        str_buf += str(im_size[0]*(h_skip + c) + w_skip) + ' ' + str(im_size[0] - w_skip*2) + ' '
    return str_buf

for i in range(1, 17):
    d[i] = get_str(i)

ss['rle_mask'] = ss.apply(lambda row: d[int(row['img'].split('_')[1].split('.')[0])], axis=1)
ss.to_csv('submission.csv.gz', compression='gzip', index=False)