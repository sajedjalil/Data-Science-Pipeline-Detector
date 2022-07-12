# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import time

preds = np.zeros((8607230, 3), dtype=int)

start_time = time.time()
out = open('sample_submission_2.csv', "w")
out.write("row_id,place_id\n")
rows = ['']*1000
n=0
for num in range(0,8600001):
	rows[n]='%d,%d %d %d\n' % (num,preds[num,0],preds[num,1],preds[num,2])
	n = n+1
	if num%1000 == 0:
		out.writelines(rows)
		rows = ['']*1000
		n=0
rows = ['']*7231
n=0
for num in range(8600001,8607230):
    rows[n]='%d,%d %d %d\n' % (num,preds[num,0],preds[num,1],preds[num,2])
    n=n+1
out.writelines(rows)
out.close()
print("Elapsed time 3: %s seconds" % (time.time() - start_time))

import pandas as pd

start_time = time.time()
df_aux = pd.DataFrame(preds, dtype=str, columns=['c1', 'c2', 'c3'])
ds_sub = df_aux.c1.str.cat([df_aux.c2, df_aux.c3], sep=' ')
ds_sub.name = 'place_id'
ds_sub.to_csv('submission_sample3.csv', index=True, header=True, index_label='row_id')
print("Elapsed time 4: %s seconds" % (time.time() - start_time))

start_time = time.time()
with open('sample_submission.csv', "w") as out:

    out.write("row_id,place_id\n")
    
    rows = ['']*8607230
    
    for num in range(0,8607230):
    
        rows[num]='%d,%d %d %d\n' % (num,preds[num,0],preds[num,1],preds[num,2])
    
    out.writelines(rows)

print("Elapsed time 5: %s seconds" % (time.time() - start_time))