# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

p_tab = pd.read_csv('../input/producto_tabla.csv')
print(p_tab.shape)
print(p_tab.head(10))

empty = np.array([None]*p_tab.shape[0])
newCols = ['desc', 'pct', 'other', 'inlabel', 'p', 'pq', 'g', 'ml', 'code1', 'code2', 'brand']
for col in newCols:
    p_tab[col] = empty

# seperate features from product list

for i in range(p_tab.shape[0]):
    p_lab = p_tab.iloc[i].NombreProducto
    #p_id_tab = p_tab.iloc[i].Producto_ID
    
    desc, pct, other, inlabel, p, pq, g, ml, code1, code2, brand, p_id = [None]*12
    descRE = re.search('([A-Za-z\s+]+)(\d+pct)\s+([A-Za-z\s+]+)', p_lab)
    if descRE:
        desc, pct, other = descRE.groups()
    else:
        desc = re.search('([A-Za-z\s+]+)', p_lab).group(1)
        pctRE = re.search('(\d+pct)\s', p_lab)
        if pctRE:
            pct = pctRE.group(1)
    
    inlabelRE = re.search('(\d+in)', p_lab)
    if inlabelRE:
        inlabel = inlabelRE.group(1)
    pRE = re.search('(\d+)p\s', p_lab)
    if pRE:
        p = pRE.group(1)
    pqRE = re.search('(\d+)pq\s', p_lab)
    if pqRE:
        pq = pqRE.group(1)
    gRE = re.search('(\d+)g\s', p_lab)
    if gRE:
        g = gRE.group(1)
    mlRE = re.search('(\d+)ml\s', p_lab)
    if mlRE:
        ml = mlRE.group(1)
    codeRE = re.search('([0-9]+[inpg]\s+)([A-Za-z]+\s+)([A-Za-z]+\s+)([A-Z]+\s+)(\d+)', p_lab)
    if codeRE:
        code1, code2, brand, p_id = codeRE.groups()[-4:]
    else:
        codeRE = re.search('([0-9]+[inpg]+\s+)+([A-Za-z]+\s+)([A-Z]+\s+)(\d+)', p_lab)
        if codeRE:
            code1, brand, p_id = codeRE.groups()[-3:]
        else:
            codeRE = re.search('([0-9]+[inpg]+\s+)+([A-Z]+\s+)(\d+)', p_lab)
            if codeRE:
                brand, p_id = codeRE.groups()[-2:]
    
    
    for col, var in zip(newCols, [desc, pct, other, inlabel, p, pq, g, ml, code1, code2, brand, p_id]):
        #print(col, var)
        p_tab.loc[i, col] = var
    
    #print(p_tab.iloc[i])
    
    if i%250==0:
        print(str(int(float(i)/p_tab.shape[0]*100)) + "% complete")
    
print('finished')

p_tab.to_hdf('product_table_all_feats.h5', 'table', append=True)

print(p_tab.shape)

print(p_tab.head(50))