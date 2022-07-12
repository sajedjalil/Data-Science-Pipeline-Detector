import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

file_name = 'test'
df = pd.read_csv('../input/'+file_name+'.csv') #nrows=30000
tabla_merete = df.shape[0] # sorok szama
print('A '+ file_name + ' halmazban szereplo rekordok szama: ' + str(tabla_merete)) # The script was killed, likely for trying to exceed the memory limit of 8192M
#print(df)
#---------------------------
oszlopok_szama = df.shape[1] #oszlopok szama
#minden oszlopra keszul egy ideiglenes tabla, ami nem tartalmazza a null ertekeket
#ennek az ideiglenes tablanak megnezzuk a meretet es kivonjuk az eredetibol ezt,
#igy megkapjuk hogy mennyi null ertek volt
for i in range(1,oszlopok_szama):
    tmp_df = pd.notnull(df.iloc[:, i])
    nem_nulla_ertekek_szama = tmp_df.shape[0]
    print('A '+ file_name + ' halmazban szereplo ' + tmp_df.name + ' nevu oszlopban:\nnull ertekek szama: ' + str(tabla_merete - tmp_df.shape[0]))
    tmp_df = df.iloc[:, i]
    if(tmp_df.dtype == 'int64'):
        print('maximum erteke: ' + str(tmp_df.max()))
        print('minimum erteke: ' + str(tmp_df.min()))
        print('atlag erteke: ' + str(tmp_df.mean()))
        print('median erteke: ' + str(tmp_df.median()))
        print('szoras erteke: ' + str(tmp_df.std()) + '\n')
    if(tmp_df.dtype == 'object'):
        print('ertekek szama: ' + str(nem_nulla_ertekek_szama))
        print('leggyakoribb ertek: ' + str(tmp_df.value_counts().idxmax()) + '\n')
print('')
#---------------------------
#client_id_df = df.iloc[:, (4,5,10)]
#print(tmp_df)
#adjusted_demand = []
#client_id_array = []
#product_id_array = []
#client_id = 0
#product_id = 0
#if list.contains(myItem):
#for i in range(0,tabla_merete):
#    
#    if(!client_id_array.contains(client_id) && !(product_id_array.contains(product_id)))
#        
#
#for i in range(0,tabla_merete):
#    print(adjusted_demand[i])