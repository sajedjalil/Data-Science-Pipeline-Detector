import json
from sklearn import svm
from sklearn.svm import SVC

def main(train):

	#open training set
	with open(train) as data_file:
		data = json.load(data_file)

	#initialize list for ingredients cmposing of all ingredients
	ingrList = []
	for i in range(len(data)):
		ingredients = data[i]['ingredients']
		ingrList = ingrList + ingredients
	
	#turn list into set so there are only unique ingredients, revert to list
	ingrSet = set(ingrList)
	UingrList = list(ingrSet)
	
	
	#create the x and y arrays; x = recipes, y = ingredients
	#len(x) == len(y), but len(y[i]) == len(UingrList)
	I = []
	C = []

	#put the values into I and C. 
	for i in range(len(data)):
		I.append([])
		C.append(data[i]['cuisine'])
		for j in range(len(UingrList)):
			if UingrList[j] in data[i]['ingredients']:
				I[i].append(1)
			else:
				I[i].append(0)

	#insert values into SVM model
	#clf = svm.SVC()
	#clf.fit(I, C)

	#instead, insert values into linear_SVM model
	lin_clf = svm.LinearSVC()
	lin_clf.fit(I, C)


	#repeat above steps for test set, except not fitting data and use IDs instead
	with open('test.json') as test_file:
		test = json.load(test_file)

	IDS = []
	INGR = []

	for i in range(len(test)):
		INGR.append([])
		IDS.append(test[i]['id'])
		for j in range(len(UingrList)):
			if UingrList[j] in test[i]['ingredients']:
				INGR[i].append(1)
			else:
				INGR[i].append(0)

	#instead of fitting, put the ingredient array into predict to get prediction
	#result = clf.predict(INGR)

	#instead of using clf, use linearclf to predict
	result = lin_clf.predict(INGR)

	#write results
	f = open("svm.csv", "w")
	f.write("id,cuisine\n")
	for i in range(len(test)):
		id = IDS[i]
		cuisine = result[i]
		f.write("%d,%s\n" % (id,cuisine))
	f.close()

def execute():
	main('train.json')