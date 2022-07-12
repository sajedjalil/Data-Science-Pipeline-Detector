##########################################################################################################
############################################            ##################################################
############################################ INITIALIZE ##################################################
############################################            ##################################################
##########################################################################################################

import numpy as np
import pandas as pd
import os

# Leemos datos
print(os.listdir('../input/'))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Nos aseguramos que no hay missings
print(train.isnull().values.any())
print(test.isnull().values.any())

# Definimos dataframe de outputs solo con ID
outputs=pd.DataFrame(test.ID_code)


##########################################################################################################
#############################################          ###################################################
############################################# COMPRESS ###################################################
#############################################          ###################################################
##########################################################################################################

INT8_MIN    = np.iinfo(np.int8).min
INT8_MAX    = np.iinfo(np.int8).max
INT16_MIN   = np.iinfo(np.int16).min
INT16_MAX   = np.iinfo(np.int16).max
INT32_MIN   = np.iinfo(np.int32).min
INT32_MAX   = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max

def memory_usage(data, detail=1):
    if detail:
        display(data.memory_usage())
    memory = data.memory_usage().sum() / (1024*1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory

def compress_dataset(data):
    """
        Compress datatype as small as it can
        Parameters
        ----------
        path: pandas Dataframe

        Returns
        -------
            None
    """
    memory_before_compress = memory_usage(data, 0)
    print()
    length_interval      = 50
    length_float_decimal = 4

    print('='*length_interval)
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != 'object':
            print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == 'float64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))))
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))
                    print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))
                    print("compress float64 --> float32")
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                print('='*length_interval)

            if col_dtype == 'int64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))
                type_flag = 64
                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                if type_flag == 32:
                    print("compress (int64) ==> (int32)")
                elif type_flag == 16:
                    print("compress (int64) ==> (int16)")
                else:
                    print("compress (int64) ==> (int8)")
                print('='*length_interval)

    print()
    memory_after_compress = memory_usage(data, 0)
    print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))

# Unimos
data = train.append(test)

# Compress
compress_dataset(data)



# Cálculo del WOE
###################################################################################
# Malos y Buenos totales
BAD=sum(train.target)
GOOD=len(train)-BAD

# Función de cálculo de WOE sobre un dataset (ds) y columna NUMÉRICA
# El dataset debe tener parte train (con target informado) y parte test con target Null
# Devuelve el mismo dataset pero con la columna transformada (y su nombre acabado en _W)
def WoE_num(ds,i_col,n_buckets):
    ds['bucket']=pd.qcut(ds[i_col], q=n_buckets,labels=False,duplicates='drop')
    tabla_woe=ds[['bucket','target']].groupby(['bucket']).sum(skipna=True).reset_index()

    if 0 in tabla_woe['target'].values:
        ds['bucket']=pd.qcut(ds[i_col], q=n_buckets-1,labels=False,duplicates='drop')
        tabla_woe=ds[['bucket','target']].groupby(['bucket']).sum(skipna=True).reset_index()
# La bucketización en Python está bien hecha (he visto que el "ntile" de R parte los empates
# como le sale de los huevos.
# Nos aseguramos que al tirar los cortes repetidos con "duplicates='drop'" (esto pasa cuando
# una variable tiene acumulación de valores repetidos) almenos queden 5 buckets restantes.
# Si no, no woeizamos la variable.
    if len(ds['bucket'].unique())>=5:
        tabla_woe = tabla_woe.rename(columns={'target': 'BAD'})
        tabla_woe['TOTAL']=ds[['bucket','target']].groupby(['bucket']).count().reset_index()['target']
        tabla_woe['GOOD']=(tabla_woe['TOTAL']-tabla_woe['BAD']).astype(int)
        # Cálculo WOE por bucket
        tabla_woe['WOE']=np.log((tabla_woe['GOOD']/GOOD)/(tabla_woe['BAD']/BAD))
        # Nueva variable "WOEizada"
        ds=pd.merge(ds, tabla_woe[['bucket','WOE']], on='bucket', how='left')
        # Gestión de nombres
        ds = ds.drop('bucket', axis=1)
        ds = ds.rename(columns={'WOE': i_col+"_W"})
        # Eliminamos variable original
        ds = ds.drop(i_col, axis=1)
    else:
        ds = ds.drop(i_col, axis=1)
        ds = ds.drop('bucket', axis=1)
    return(ds)

# Lista con las variables a las que vamos a "WOEizar"
predictoras=list(set(list(train))-set(['ID_code','target']))
# Bucle sobre las variables
data_woe=data
i=1
for nombre_columna in predictoras:
    print(nombre_columna,"...",i,"de",len(predictoras))
    data_woe=WoE_num(data_woe,nombre_columna,20)
    i=i+1

# Separamos
train=data_woe[data_woe['ID_code'].str.contains("train")]
test=data_woe[data_woe['ID_code'].str.contains("test")]
test=test.drop('target',axis=1)

##########################################################################################################
################################################         #################################################
################################################ MODELOS #################################################
################################################         #################################################
##########################################################################################################

################################################################################
# Toda la información aquí: "https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/"
################################################################################
import catboost as cat

# 1) Definimos conjuntos de entrenamiento y test:
################################################################################
predictoras=list(set(list(train))-set(['ID_code','target']))
X_train=train[predictoras].reset_index(drop=True)
Y_train=train.target.reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# Partición conjunto de entrenamiento y validación
RE=12345 # Seed que utilizaremos para la partición y la parte random del modelo
TS=0.3 # Tamaño del conjunto de validación
esr=200

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=RE)

# Pasamos a clase Pool
pool=cat.Pool(X_train, Y_train)
pool_tr=cat.Pool(x_tr, y_tr)
pool_val=cat.Pool(x_val, y_val)

# 2) Primer modelo CatBoost con conjunto de validación para obtener rondas óptimas.
# Aquí deberemos tunear los parámetros "a mano". Hacer un grid-search es demasiado costoso.
# Vamos haciendo pruebas con valores razonables hasta encontrar unos que funcionen bien.
# Parámetros del booster. Ver todoas las opciones en:
# Python: "https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/"
# R: "https://tech.yandex.com/catboost/doc/dg/concepts/r-training-parameters-docpage/"
################################################################################
model_catboost_val = cat.CatBoostClassifier(
          eval_metric='AUC',
          objective='Logloss',
          iterations=200000, # Valor muy alto, para encontrar el mejor modelo
          od_type='Iter',
          learning_rate=0.01, # En sintonía al número de árboles (iterations). Si no acaba por "esr", bajarlo
          depth=6, # Profundidad de los árboles (poner valores entre 5 y 10, más alto -> más overfitting)
          l2_leaf_reg=10, # Regularización L2 (poner entre 3 y 20, más alto -> menos overfitting).
          rsm=0.5, # % de features para hacer cada split (más bajo: acelera la ejecución y reduce overfitting)
          random_seed=RE,
          verbose=1000) # Nos muestra la métrica train/test cada tantos árboles

print('\nCatBoost Fit (Validation)...\n')
# Para ir más rápido solo evaluamos cada 1000 rondas:
model_catboost_val.fit(X=pool_tr,
                         eval_set=pool_val,
                         use_best_model=True,
                         early_stopping_rounds=esr)

# Obtenemos número óptimo de rondas
best_nrounds=int((model_catboost_val.get_best_iteration())/(1-TS))
print(best_nrounds)

# 3) Modelo sobre todo el train con los parámetros óptimos y número de rondas óptimas de la validación
################################################################################
model_catboost = cat.CatBoostClassifier(
          objective='Logloss',
          iterations=best_nrounds,
          learning_rate=0.01,
          depth=6,
          l2_leaf_reg=10,
          rsm=0.5,
          random_seed=RE,
          verbose=1000)

print('\nCatBoost Fit...\n')
model_catboost.fit(X=pool,metric_period=1000)


##########################################################################################################
###############################################            ###############################################
############################################### RESULTADOS ###############################################
###############################################            ###############################################
##########################################################################################################

# Predicción final (submission). Hacemos exp pq la submission es con el precio
################################################################################
test['target']=model_catboost.predict_proba(X_test)[:,1]
outputs_catboost=pd.merge(outputs, test[['ID_code','target']], on='ID_code', how='left')

# Outputs a .csv
################################################################################
outputs_catboost.to_csv('outputs_catboost_01_woe_data.csv', index = False)
print('\nEND')

# Códigos por si queremos salvar/recuperar un modelo largo de ejecutar
################################################################################
# Ejemplo de como salvar un modelo:
# from sklearn.externals import joblib
# joblib.dump(model_catboost,'model_catboost_01.sav')

# Como cargarlo y hacer predicciones
# loaded_model=joblib.load('BBDD Output/model_catboost.sav')
# loaded_model.predict(X1_test)
################################################################################

##########################################################################################################
#############################################                #############################################
############################################# FIN DE LA CITA #############################################
#############################################                #############################################
##########################################################################################################
