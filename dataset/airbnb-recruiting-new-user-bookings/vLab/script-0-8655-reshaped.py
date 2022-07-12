import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)


#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]
    

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)


#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
tmp = list()
for x in dac:
    date = datetime.datetime(x[0], x[1], x[2])
    tmp.append((x[0], x[1], x[2], date.weekday(), int(date.timestamp())))
dac = np.array(tmp)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all['dac_day_in_week'] = dac[:,3]
df_all['dac_timestamp'] = dac[:,4]

df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
tmp = list()
for x in tfa:
    date = datetime.datetime(x[0], x[1], x[2])
    tmp.append((x[0], x[1], x[2], date.weekday(), int(date.timestamp())))
tfa = np.array(tmp)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all['tfa_day_in_week'] = tfa[:,3]
df_all['tfa_timestamp'] = tfa[:,4]

df_all = df_all.drop(['timestamp_first_active'], axis=1)


#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)
#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]
#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(X, y, eval_metric='ndcg@5')
y_pred = xgb.predict_proba(X_test)  
from math import log2

def DCG(liste):
	ll = [(2**rel-1)/(log2(index+2)) for index,rel in enumerate(liste)]
	return sum(ll)

def nDCG(liste):
	IDCG = DCG(sorted(liste, reverse=True));
	if IDCG == 0:
		return 0
	else:
		return DCG(liste) / IDCG

# size est la taille que doivent respecter toutes les listes "mask" passées en paramètre
# masks est une liste de mask (= liste contenant des valeurs 0 ou 1) de même taille, ordonnée 
# proba est une distribution de probabilité ordonnée
# retourne un couple de data : 
# 1) le "vecteur réponse" : de taille = 5, il présente le choix fait. ex :[1,2,2,3,3]
# 2) le score du nDCG associé
def apply(size, masks, proba):
	score = 0
	combi = [0 for n in range(size)]
	for i, mask in enumerate(masks):
		score += nDCG(mask) * proba[i]
		for j, val in enumerate(mask):
			if val == 1:
				combi[j] = i
	return combi, score

def combi1(proba):
	return apply(5, [[1,0,0,0,0], [0,1,1,1,1]], proba)

def combi2(proba):
	return apply(5, [[1,0,0,0,0], [0,1,1,1,0], [0,0,0,0,1]], proba)
	
def combi3(proba):
	return apply(5, [[1,0,0,0,0], [0,1,1,0,0], [0,0,0,1,1]], proba)
	
def combi4(proba):
	return apply(5, [[1,0,0,0,0], [0,1,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], proba)

def combi5(proba):
	return apply(5, [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,1,1]], proba)

def combi6(proba):
	return apply(5, [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,1,0], [0,0,0,0,1]], proba)
	
def combi7(proba):
	return apply(5, [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,1]], proba)
	
def combi8(proba):
	return apply(5, [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], proba)

combis = [combi1, combi2, combi3, combi4, combi5, combi6, combi7, combi8]

def getBestCombi(proba):
	bestScore = 0.0
	bestCombi = []
	for combi in combis:
		vecteur, score = combi(proba)
#		print(str(vecteur) + " " + str(score))
		if score > bestScore:
			bestScore = score
			bestCombi = vecteur
	return bestCombi
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
#    idx = id_test[i]
#    ids += [idx] * 5
#   # selectedSortedDest
#    ssd = np.argsort(y_pred[i])[::-1][:5]
#    selectedSortedProba = np.sort(y_pred[i])[::-1][:5]
#    bestCombi = getBestCombi(selectedSortedProba)
#    finalCombi = [ssd[bestCombi[0]], ssd[bestCombi[1]], ssd[bestCombi[2]], ssd[bestCombi[3]], ssd[bestCombi[4]]]
#    cts += le.inverse_transform(finalCombi).tolist()    

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)