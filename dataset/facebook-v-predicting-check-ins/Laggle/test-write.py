import numpy as np

preds = np.zeros((10, 3), dtype=int)

f = open('sub_knn.csv','w')
f.write('row_id, place_id\n')
for row in range(0, len(preds)):
    f.write(str(row) + ',' + str(preds[row][0]) + ' ' + str(preds[row][1]) + ' ' + str(preds[row][2]) + '\n')
f.close()