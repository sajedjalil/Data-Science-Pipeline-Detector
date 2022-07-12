from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.decomposition import RandomizedPCA

train = pd.read_json('../input/train.json')
test  = pd.read_json("../input/test.json")
#sample_submission = pd.read_csv("../input/sample_submission.csv")
ingredients = []
testIngredients = []

cuisines = train['cuisine']

unique_cuisines = cuisines.unique()#不同國家


#不同的國家:20
#print(unique_cuisines)


#for unique_cuisine in unique_cuisines:
#    print(unique_cuisine)

for ingredient in train['ingredients']:
    ingredients.extend(ingredient)
    #for z in ingredient:
    #    print(z)
    #print(ingredient)

for ingredient in test['ingredients']:
    testIngredients.extend(ingredient)

print("ingredients len:" + str(len(ingredients)))


unique_ingredients = set(ingredients)

#for a in unique_ingredients:
    #print(a)

unique_testIngredients = set(testIngredients)

c = Counter(ingredients)

unique_ingredients_counter = sorted(c.items(),key=itemgetter(1), reverse=False)

c = Counter(testIngredients)

unique_testIngredients_counter = sorted(c.items(),key=itemgetter(1), reverse=False)


def printCuisinesCount(cuisines):
    unique_cuisines_c = []
    for cuisine in cuisines:
        unique_cuisines_c.append(cuisine)
    c = Counter(unique_cuisines_c)
    unique_cuisines_counter = sorted(c.items(),key=itemgetter(1), reverse=False)
    
    print("cuisines from train data: ")
    for cuisine in unique_cuisines_counter:
        print(cuisine)
    print(" ")
    return
    
printCuisinesCount(cuisines)

#"""
data = {'id':[],'cuisine':[],'ingredients':[]}
newTrain = pd.DataFrame(data)

numbercount = []

data = {'id':[],'cuisine':[],'ingredients':[]}
newTrain = pd.DataFrame(data)

#print(unique_ingredients_counter)
#print(train["ingredients"])

trashhold = 467


unique_cuisines_list = ['brazilian','russian','jamaican','irish','filipino','british','moroccan','vietnamese','korean','spanish','greek','japanese','thai','cajun_creole','french','chinese','indian','southern_us','mexican','italian']
cuisines_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def getIndex(value):
    return unique_cuisines_list.index(value) - 1

def isInthelist(container ,value):
    boolList = []
    #print(container)
    #print(value)
    for item in container:
        #print(value)
        #print(item)
        if value in item:
            boolList.append(True)
        else:
            boolList.append(False)
    #print(boolList)
    return boolList
    
def getboolListTrueCount(l):
    count=0
    for item in l:
        if item == True:
            count+=1
    return count

def OrList(l1,l2):
    for i in range(0,len(l1),1):
        if l2[i] and (not l1[i]):
            index = getIndex(train["cuisine"].loc[i])
            #print(cuisines_count[index])
            if cuisines_count[index] < trashhold:
                l1[i] = l2[i]
                cuisines_count[index] +=1
    return l1;

boolList = isInthelist(train["ingredients"],"");

for z in unique_ingredients_counter:
    boolList = OrList(boolList ,isInthelist(train["ingredients"],z[0]))
    num = getboolListTrueCount(boolList)
    if num > (trashhold *20):
        break
newTrain = train[boolList]
print(len(newTrain.index))
print(getboolListTrueCount(boolList))
print(cuisines_count)
c = Counter(numbercount)
numbercount = sorted(c.items(),key=itemgetter(1), reverse=False)
newIngredients = []
for ingredient in newTrain['ingredients']:
    newIngredients.extend(ingredient)
    
c = Counter(newIngredients)

unique_ingredients_counter = sorted(c.items(),key=itemgetter(1), reverse=False)

for z in unique_ingredients_counter:
    print(z)
#for z in numbercount:
    #print(z)
#"""

printCuisinesCount(newTrain["cuisine"])

#for index, row in newTrain.iterrows():
    #print(row['ingredients'])

print(" ")

#pca = RandomizedPCA(n_components=2)
#    X_pca = pca.fit_transform(X)

print("unique ingredients from train:" + str(len(unique_ingredients)))
print("unique ingredients from test:" + str(len(unique_testIngredients)))
#不同食材:6714


print("test ingredinet not in train: ")
diff = 0
for ingredient in unique_testIngredients:
    if ingredient not in unique_ingredients:
        print(ingredient)
        diff +=1
        
print(diff)

#print(unique_ingredients_clean_string)
#print(ingredients_clean_string)

