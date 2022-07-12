# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:39:13 2016

@author: Administrator
"""

import pandas as pd
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
               'ind_cder_fin_ult1', 'ind_cno_fin_ult1' , 'ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
               'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
               'ind_nomina_ult1'  , 'ind_nom_pens_ult1', 'ind_recibo_ult1']

#值映射字�?
mapping_dict = {'ind_empleado' : {-99:0, 'A':1, 'B':2, 'F':3, 'N':4, 'S':5},
                'indrel_1mes'  : {-99:0, 1.0:1, '1.0':1 ,'1':1, 2.0:2, '2.0':2, '2':2, '3.0':3, 3.0:3, '3':3, '4.0':4, 4.0:4, '4':4, 'P':5},
                'tiprel_1mes'  : {-99:0, 'A':1, 'I':2, 'P':3, 'R':4, 'N':5},
                'segmento'     : {-99:0, '01 - TOP':1, '02 - PARTICULARES':2, '03 - UNIVERSITARIO':3},
                'ind_nuevo'    : {-99:0, 0.0:2, 1.0:1},
                'indrel'       : {-99:0, 99:2, 1.0:1},
                'indresi'      : {-99:0, 'S':1, 'N':2},
                'indext'       : {-99:0, 'S':1, 'N':2},
                'indfall'      : {-99:0, 'S':1, 'N':2},
                'sexo'         : {-99:0, 'V':1, 'H':2},
                'conyuemp'     : {-99:0, 'S':1, 'N':2},
                'ind_actividad_cliente':{-99:0, 0.0:2, 1.0:1}
                }
               
#my_columns = ['ind_empleado','sexo','age','ind_nuevo','antiguedad','indrel',
#             'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp',
#             'indfall','ind_actividad_cliente','segmento'
#             ]
             
my_columns = ['ind_empleado','sexo','ind_nuevo','indrel',
             'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp',
             'indfall','ind_actividad_cliente','segmento'
             ]                

print ("Start reading training file and testing file.....")
columns_need_mapping = list(mapping_dict.keys())            
#df_train = pd.read_csv("../input/train_ver2.csv", usecols=my_columns+target_cols)
#df_test = pd.read_csv("../input/test_ver2.csv", usecols=my_columns)
#df_target = pd.read_csv("../input/train_ver2.csv", usecols=target_cols)

df_train = pd.read_csv("../input/train_ver2.csv", usecols=my_columns)
df_test = pd.read_csv("../input/test_ver2.csv", usecols=my_columns)
df_target = pd.read_csv("../input/train_ver2.csv", usecols=target_cols)
df_ncodpers_test = pd.read_csv("../input/test_ver2.csv",usecols=['ncodpers'])
df_dato_train = pd.read_csv("../input/train_ver2.csv", usecols=['ncodpers','fecha_dato'])

#df_train = pd.read_csv("train_sub.csv", usecols=my_columns)
#df_test = pd.read_csv("test_sub.csv", usecols=my_columns)
#df_target = pd.read_csv("target.csv", usecols=target_cols)
#df_ncodpers_test = pd.read_csv("test_sub.csv",usecols=['ncodpers'])
#df_dato_train = pd.read_csv("train_sub.csv", usecols=['ncodpers','fecha_dato'])

print ("start cleaning the dataset........")
df_train.fillna(-99, inplace=True)
df_test.fillna(-99, inplace=True)
df_target.fillna(0, inplace=True)

#数据清洗
for col in columns_need_mapping:
    df_train[col] = df_train[col].apply(lambda x: mapping_dict[col][x])
    df_test[col] = df_test[col].apply(lambda x: mapping_dict[col][x])

print ("start training and predicting.......")
for col in target_cols:
    df_train[col]=df_target[col]
    df_train.drop_duplicates(keep='first', inplace=True)
    
    rf = RandomForestClassifier()
    rf.fit(df_train.drop(col, axis=1), df_train[col])
    
    #score=rf.score(df_train.drop(col, axis=1), df_train[col])
        
    retvalue = rf.predict(df_test)
    df_ncodpers_test[col]=retvalue
    df_train.drop(col, axis=1, inplace=True)
    
print ("Get the products already for each ncodper.....")
for col in target_cols:
    df_dato_train[col]=df_target[col]

df_dato_train = df_dato_train[df_dato_train['fecha_dato']=='2016-05-28']

print ("analysing........")
submission_dict = dict()
for user_id in df_ncodpers_test['ncodpers']:
    add_product=list()
    test_tmp = df_ncodpers_test[df_ncodpers_test['ncodpers']==user_id]
    already_tmp = df_dato_train[df_dato_train['ncodpers']==user_id]    
    for col in target_cols:
        if((test_tmp.iat[0, target_cols.index(col)+1])>(already_tmp.iat[0, target_cols.index(col)+2])):
            add_product.append(col)
            
    submission_dict[user_id]=(' '.join(add_product))
#    new=pd.DataFrame([dict(ncodpers=user_id, add_products=' '.join(add_product))])
#    submission=submission.append(new, ignore_index=True)

print ("outputing......")
submission = pd.DataFrame(submission_dict.items(),columns=['ncodpers','add_products'])
submission.to_csv('sub.csv', index=False, columns=['ncodpers','added_products'])

print ("ending")

##print "start PCA......"
###PCA降维
##pca = PCA(n_components='mle')
##new_data = pca.fit(df_train)