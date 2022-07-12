##########################################################################################################
############################################            ##################################################
############################################ INITIALIZE ##################################################
############################################            ##################################################
##########################################################################################################

import numpy as np
import pandas as pd

train = pd.read_csv('../input/newtrain/newtrain.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

train=train.drop('pred',axis=1)

# Nos aseguramos que no hay missings
print(train.isnull().values.any())
print(test.isnull().values.any())

# Definimos dataframe de outputs solo con ID
outputs=pd.DataFrame(test.ID_code)


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
RE=54321 # Seed que utilizaremos para la partición y la parte random del modelo
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
          verbose=500) # Nos muestra la métrica train/test cada tantos árboles

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
outputs_catboost.to_csv('outputs_catboost_01_newtrain.csv', index = False)
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
