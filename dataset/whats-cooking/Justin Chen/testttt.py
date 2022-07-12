'''
k nearest neighbor model
'''
import json as js
import random as rd
import numpy as np

def Jacarrd(lista, listb):
    seta = set(lista)
    setb = set(listb)
    return float(len(seta&setb))/float(len(seta|setb))

with open('../input/train.json') as training_data_file:
    training_data = js.load(training_data_file)

with open('../input/test.json') as test_data_file:
    test_data = js.load(test_data_file)

#----- prepare the cuisine/ingredient index dictionary --------
n_training = len(training_data)
n_test = len(test_data)

cuisine_list = list()
ingredient_list = list()

for i in range(n_training):
    if training_data[i]['cuisine'] not in cuisine_list:
        cuisine_list.append(training_data[i]['cuisine'])
    for j in training_data[i]['ingredients']:
        if j not in ingredient_list:
            ingredient_list.append(j)

cuisine_dict = dict()
ingredient_dict = dict()

for i in range(len(cuisine_list)):
    cuisine_dict[cuisine_list[i]] = i
for i in range(len(ingredient_list)):
    ingredient_dict[ingredient_list[i]] = i

# ------   KNN -----------
file_out = open('knnoutput.csv','w')
file_out.write('id,cuisine\n')

nk = 5 # the user defined K
n_sample = 500
d = np.zeros([20])
for t in range(n_test):
    d = np.zeros([n_training])
    cuisine = np.zeros([20])
    for s in range(n_sample):
        j = rd.choice(range(n_training))
        d[j] = Jacarrd(test_data[t]['ingredients'], training_data[j]['ingredients'])
    for k in range(nk):
        maxid = np.argmax(d)
        cuisine[cuisine_dict[training_data[maxid]['cuisine']]] += 1.0
        d[maxid] = -10.0
    prediction = np.argmax(cuisine)

    file_out.write(str(test_data[t]['id']))
    file_out.write(',')
    file_out.write(cuisine_list[prediction])
    file_out.write('\n')

file_out.close()