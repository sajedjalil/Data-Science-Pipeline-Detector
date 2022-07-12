import numpy as np
import pandas as pd

im_size = [1918, 1280]
w_skip = 500
h_skip = 430

str_buf = ''
for c in range(im_size[1] - w_skip*2):
	str_buf += str(im_size[0]*(w_skip + c) + 430) + ' ' + str(im_size[0] - h_skip*2) + ' '

print(str_buf)

ss = pd.read_csv('../input/sample_submission.csv')
ss['rle_mask'] = str_buf
ss.to_csv('submission.csv.gz', compression = "gzip", index=False)