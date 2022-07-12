import pandas

trainingData = pandas.read_json("../input/train.json")

#print(pandas.value_counts(trainingData['cuisine']))

cuisine_mapping = {'italian' : 0, 'mexican' : 1, 'southern_us': 2, 'indian' : 3, 'chinese' : 4, 'french' : 5, 'cajun_creole' : 6, 'thai' : 7,
    'japanese' : 8, 'greek' : 9, 'spanish' : 10, 'korean' : 11, 'vietnamese' : 12, 'moroccan' : 13, 'british': 14, 'filipino':15, 'irish':16,
    'jamaican' : 17, 'russian' :18, 'brazilian' : 19}

for k, v in cuisine_mapping.items():
    trainingData.loc[trainingData['cuisine'] == k,'cuisine'] = v

#print(pandas.value_counts(trainingData['cuisine']))
print(trainingData.head(5))

print(type(trainingData['cuisine']))