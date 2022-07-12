import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def readFile(fileName):
	return pd.read_csv(fileName)
train = readFile('../input/train.csv')
test = readFile('../input/test.csv')

def removeIDTarget(train, test):
	X = train.drop(['TARGET','ID'], axis=1)
	y = train['TARGET']
	X_test = test.drop(['ID'], axis=1)
	y_test = test.ID
	print(X.shape)
	print(X_test.shape)
	return (X,y,X_test,y_test)

X, y, test, test_index = removeIDTarget(train, test)
	
def removeUniqueColumn(train, test):
	remove = []
	for col in train.columns:
		if train[col].std() == 0:
			remove.append(col)
	train = train.drop(remove, axis=1)
	test = test.drop(remove, axis=1)
	return (train, test)

X, test = removeUniqueColumn(X, test)

from sklearn.preprocessing import normalize
numpyMatrix = normalize(X, norm='l2', axis=1)
print(numpyMatrix.shape)

def transformAsMatrix(X):
	return X.as_matrix()
	
numpyMatrix = transformAsMatrix(X)
y = transformAsMatrix(y)



#------------------------------------------------------
from sklearn import decomposition
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def pca_traitement(numpyMatrix, test):
	pca = decomposition.PCA()
	pca.fit(numpyMatrix)
	pca.n_components = 50
	train_new = pca.fit_transform(numpyMatrix)
	test_new = pca.transform(test)
	return (train_new,test_new)
X_new_pca, test_new_pca = pca_traitement(numpyMatrix, test)
print("X_new_pca: ", X_new_pca.shape)
print("test_new_pca:", test_new_pca.shape)

def varianceThreshold(numpyMatrix, test):
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	train_new = sel.fit_transform(numpyMatrix)
	test_new = sel.transform(test)
	return (train_new, test_new)
#X_new_vt, test_new_vt = varianceThreshold(numpyMatrix, test)
#print("X_new_vt: ", X_new_vt.shape)
#print("test_new_vt:", test_new_vt.shape)

def linearSVC(numpyMatrix, test):
	lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(numpyMatrix,y)
	model = SelectFromModel(lsvc, prefit=True)
	train_new = model.transform(numpyMatrix)
	test_new = model.transform(test)
	return (train_new, test_new)
#X_new_svc, test_new_svc = linearSVC(numpyMatrix, test)
#print("X_new_svc: ", X_new_svc.shape)
#print("test_new_svc:", test_new_svc.shape)

def extraTreesClassifier(numpyMatrix, y, test):
	clf = ExtraTreesClassifier()
	clf = clf.fit(numpyMatrix,y)
	clf.feature_importances_

	model = SelectFromModel(clf, prefit=True)
	train_new = model.transform(numpyMatrix)
	test_new = model.transform(test)
	return (train_new,test_new)
X_new_etc, test_new_etc = extraTreesClassifier(numpyMatrix, y, test)
print("X_new_etc: ", X_new_etc.shape)
print("test_new_etc:", test_new_etc.shape)
	
#-------------------------predict_proba--------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
#--------------------------start execution time------------------
import time
start_time = time.time()

def predict_prob(type_pred, X_train, y_train, X_test):
	result = []
	if type_pred == 1: #KNN
		neigh = KNeighborsClassifier(n_neighbors=500)
		neigh.fit(X_train, y_train)
		for i in range(len(X_test)):
			r = neigh.predict_proba([X_test[i]])[0]
			result.append(r[1])
	if type_pred == 2: #BernoulliNB
		bnb = BernoulliNB()
		bnb.fit(X_train, y_train)
		for i in range(len(X_test)):
			r = bnb.predict_proba([X_test[i]])[0]
			result.append(r[1])
	if type_pred == 3: #Decision tree
		dt = tree.DecisionTreeClassifier()
		dt = dt.fit(X_train, y_train)
		result = []
		for i in range(len(X_test)):
			r = dt.predict_proba([X_test[i]])[0]
			result.append(r[1])
	return result

#----------------------------validation-----------------------------
def validation(type_pred, X_train, y_train, X_test):
	result = []
	if type_pred == 1: #KNN
		neigh = KNeighborsClassifier(n_neighbors=500)
		neigh.fit(X_train, y_train)
		for i in range(len(X_test)):
			r = neigh.predict([X_test[i]])[0]
			result.append(r)
	if type_pred == 2: #BernoulliNB
		bnb = BernoulliNB()
		bnb.fit(X_train, y_train)
		for i in range(len(X_test)):
			r = bnb.predict([X_test[i]])[0]
			result.append(r)
	if type_pred == 3: #Decision tree
		dt = tree.DecisionTreeClassifier()
		dt = dt.fit(X_train, y_train)
		result = []
		for i in range(len(X_test)):
			r = dt.predict([X_test[i]])[0]
			result.append(r)
	return result

from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def k_fold_cross_validation(X, y, K, randomise = False):
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in range(K):
		X_train = [x for i, x in enumerate(X) if i % K != k]
		y_train = [x for i, x in enumerate(y) if i % K != k]
		X_validation = [x for i, x in enumerate(X) if i % K == k]
		y_validation = [x for i, x in enumerate(y) if i % K == k]
		yield X_train, X_validation, y_train, y_validation

#X = [i for i in range(97)]
result_valid_KNN = []
result_valid_BNB = []
result_valid_DT = []
knn = []
bnb = []
dt = []
for X_train, X_validation, y_train, y_validation in k_fold_cross_validation(X_new_pca, y, K=7):
#for X_train, X_validation, y_train, y_validation in k_fold_cross_validation(X_new_vt, y, K=7):
#for X_train, X_validation, y_train, y_validation in k_fold_cross_validation(X_new_svc, y, K=7):
#for X_train, X_validation, y_train, y_validation in k_fold_cross_validation(X_new_etc, y, K=7):
	print("round")
	'''
	knn = validation (1, X_train, y_train, X_validation)
	accuracy = accuracy_score(knn, y_validation)
	result_valid_KNN.append(accuracy)
	print("accuracy knn : ", accuracy)
	'''
	
	bnb = validation (2, X_train, y_train, X_validation)
	accuracy = accuracy_score(bnb, y_validation)
	result_valid_BNB.append(accuracy)
	print("accuracy bnb : ", accuracy)
	
	
	dt = validation (3, X_train, y_train, X_validation)
	accuracy = accuracy_score(dt, y_validation)
	result_valid_DT.append(accuracy)
	print("accuracy dt : ", accuracy)
	
	
	#cm = confusion_matrix(knn, y_validation)
	cm = confusion_matrix(bnb, y_validation)
	cm = confusion_matrix(dt, y_validation)
	

result_accuracy = []
#result_predict = []
#result_valid_KNN_mean = np.mean(result_valid_KNN)
#result_accuracy.append(result_valid_KNN_mean)
result_valid_BNB_mean = np.mean(result_valid_BNB)
result_accuracy.append(result_valid_BNB_mean)
result_valid_DT_mean = np.mean(result_valid_DT)
result_accuracy.append(result_valid_DT_mean)


#print("Mean_accuracy_KNN:", result_valid_KNN_mean)
#result11 = predict_prob(1, X_new_etc, y, X_new_etc)
#result_predict.append(result11)

print("Mean_ccuracy_BNB:", result_valid_BNB_mean)
result12 = predict_prob(2, X_new_pca, y, test_new_pca)
#result_predict.append(result12)

print("Mean_ccuracy_DT:", result_valid_DT_mean)
result13 = predict_prob(3, X_new_pca, y, test_new_pca)
#result_predict.append(result13)

print("--- %s seconds ---" % (time.time() - start_time))

#[(result_accuracy[i], result_predict) for i in range(0,3)]

#max_result = max(result_accuracy)

print("Confusion matrix: \n", cm)

from sklearn.metrics import classification_report

target_names = ["Happy","Unhappy"]
#print(classification_report(dt, y_validation, target_names=target_names))

#------------------------AUC--------------------------------
from sklearn import metrics

dt_auc = predict_prob (3, X_train, y_train, X_validation)
bnb_auc = predict_prob (2, X_train, y_train, X_validation)

print(metrics.roc_auc_score(y_validation, bnb_auc))
print(metrics.roc_auc_score(y_validation, dt_auc))

#print("Confusion matrix: \n", confusion_matrix(result_valid_BNB_moyen, y_validation))
#print("Mean_ccuracy_DT:", result_valid_DT_mean)
#print("Confusion matrix: \n", confusion_matrix(result_valid_DT_moyen, y_validation))


#write result to csv format
submission = pd.DataFrame({"ID":test_index, "TARGET":result12})
submission.to_csv("submission.csv", index=False)

print(result_accuracy)

import matplotlib.pyplot as plt

def export_graph(result_accuracy):

	fig = plt.figure()
	ax = fig.add_subplot(111)

	## the data
	N = 2

	rects = ax.patches

	## necessary variables
	ind = np.arange(N)                # the x locations for the groups
	width = 0.2                      # the width of the bars

	## the bars
	rects1 = ax.bar(ind, result_accuracy, width,
					color=['blue','orange'],
					error_kw=dict(elinewidth=2,ecolor='red'))

	ax.set_xlim(-width,len(ind))
	ax.set_ylim(0,1)
	ax.set_ylabel('Accuracy')
	ax.set_title('Comparing accuracy rates')
	xTickMarks = ['Naive Bayes', 'Decision tree']
	#xTickMarks = ['Group'+str(i) for i in range(1,6)]
	ax.set_xticks(ind+width)
	xtickNames = ax.set_xticklabels(xTickMarks)
	plt.setp(xtickNames, rotation=45, fontsize=7)

	for rect, label in zip(rects, result_accuracy):
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2, height + 1, label, ha='center', va='bottom')

	return plt.savefig("compare_accuracies.png")

export_graph(result_accuracy)