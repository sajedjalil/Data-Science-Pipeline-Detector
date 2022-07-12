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


##########################################################################################################
#############################################          ###################################################
############################################# DATSSETS ###################################################
#############################################          ###################################################
##########################################################################################################

train=data[data['ID_code'].str.contains("train")]
test=data[data['ID_code'].str.contains("test")]
test=test.drop('target',axis=1)

predictoras=list(set(list(train))-set(['ID_code','target']))

del data

##########################################################################################################
#############################################          ###################################################
############################################# MODELING ###################################################
#############################################          ###################################################
##########################################################################################################

########################################################
################## STACKING NIVEL 1 ####################
########################################################

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Definimos conjuntos de entrenamiento y test:
X_train=train[predictoras].reset_index(drop=True)
Y_train=train.target.reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# Función de entrenamiento de cada fold a través de los otros para un modelo dado
# Genera predicciones (concatenadas y libres de overfitting) a train
# Genera predicciones a test (como media de los k modelos del CV)
def Model_cv(MODEL,cat, k, X_train, X_test, y, RE):
	# Creamos los k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RE)

	# Creamos los conjuntos train y test de primer nivel
	Nivel_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	Nivel_1_test = pd.DataFrame()

	# Bucle principal para cada fold. Iniciamos contador
	count=0
	for train_index, test_index in kf.split(X_train, Y_train):
		count+=1
		# Definimos train y test en función del fold que estamos
		fold_train= X_train.loc[train_index.tolist(), :]
		fold_test=X_train.loc[test_index.tolist(), :]
		fold_ytrain=y[train_index.tolist()]
		fold_ytest=y[test_index.tolist()]

		if cat==0:
			# Ajustamos modelo con los k-1 folds
			model_fit=MODEL.fit(fold_train, fold_ytrain)

		if cat==1:
			# Preparamos pool
			pool_train=Pool(fold_train, fold_ytrain)
			# Ajustamos modelo con los k-1 folds
			model_fit=MODEL.fit(X=pool_train)

		# Predecimos sobre el fold libre para calcular el error del CV y muy importante:
		# Para hacer una prediccion a train libre de overfitting para el siguiente nivel
		p_fold=model_fit.predict_proba(fold_test)[:,1]

		# Sacamos el score de la prediccion en el fold libre
		score=roc_auc_score(fold_ytest,p_fold)
		print(k, "- cv, Fold", count, "AUC:", score)
		# Gardamos a Nivel_1_train  las predicciones "libres" concatenadas
		Nivel_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Tenemos que predecir al conjunto test para hacer la media de los k modelos
		# Definimos nombre de la predicción (p_"número de iteración")
		name = 'p_' + str(count)
		# Predicción al test real
		real_pred = model_fit.predict_proba(X_test)[:,1]
		# Ponemos nombre
		real_pred = pd.DataFrame({name:real_pred}, columns=[name])
		# Añadimos a Nivel_1_test
		Nivel_1_test=pd.concat((Nivel_1_test,real_pred),axis=1)

	# Caluclamos la métrica de la predicción total concatenada (y libre de overfitting) a train
	print("")
	print(k, "- cv, TOTAL AUC:", roc_auc_score(y,Nivel_1_train['train_yhat']))

	# Hacemos la media de las k predicciones de test
	Nivel_1_test['model']=Nivel_1_test.mean(axis=1)

	# Devolvemos los conjuntos de train y test con la predicción
	return Nivel_1_train, pd.DataFrame({'test_yhat':Nivel_1_test['model']})


# 1) Entrenamos los diferentes modelos
# Se supone que ya conocemos los parámetros óptimos (en caso que no, se tienen que testear Cross-Validados)
################################################################################

# LightGBM
from lightgbm import LGBMClassifier
print("\nCalculando LightGBM...")
LGBM_train, LGBM_test = Model_cv(LGBMClassifier(objective='binary',
											n_estimators=5761,
											learning_rate=0.01,
											num_leaves=40,
											colsample_bytree=0.5,
								            bagging_fraction=0.5,
											verbose=1000),
											0,5,X_train,X_test,Y_train,41235)


# 2) Nuevo train con las predicciones (concatenadas de los modelos cross-validados)
################################################################################
X1_train=pd.DataFrame({
					   "LGBM":LGBM_train['train_yhat']
					  })

# 3) Nuevo test con las predicciones de cada modelo (para cada modelo, es la media de los
# k submodelos surgidos de los k folds que hayamos establecido):
################################################################################
X1_test=pd.DataFrame({
					   "LGBM":LGBM_test['test_yhat']
					  })


# Guardamos para no volver a entrenar:
################################################################################
X1_train.to_csv("X1_train_LGBM.csv", index = False)
X1_test.to_csv("X1_test_LGBM.csv", index = False)
Y_train.to_csv("Y_train_LGBM.csv", index = False)