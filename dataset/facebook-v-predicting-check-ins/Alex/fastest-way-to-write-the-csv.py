# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import time

preds = np.zeros((8607230, 3), dtype=int)


start_time = time.time()
out = open('sample_submission_1.csv', "w")
out.write("row_id,place_id\n")
rows = ['']*preds.shape[0]
for num in range(0,preds.shape[0]):
	rows[num]='%d,%d %d %d\n' % (num,preds[num,0],preds[num,1],preds[num,2])
out.writelines(rows)
out.close()
print("Elapsed time 1: %s seconds" % (time.time() - start_time))