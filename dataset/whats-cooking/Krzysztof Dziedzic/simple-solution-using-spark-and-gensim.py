#load modules

import json
from gensim import corpora,models,similarities


# Reading data from json file

file=open('train.json','r')
string=file.read().strip()
data=json.loads(string)
lst=[]
for item in data:
    lst.append((item['cuisine'],item['ingredients']))


# converting data to Spark RDD

rawRDD1=sc.parallelize(lst)
rawRDD=rawRDD1.map(lambda line:(line[0],list(set(line[1]))))
uniqueCuisines=rawRDD.map(lambda x:x[0]).distinct().collect()


# Extracting Id's of features

ingredientsIds=rawRDD.flatMap(lambda x:x[1]).distinct().zipWithIndex().collectAsMap()
ingredientsIds=sc.broadcast(ingredientsIds)


# Function to parse sparse vector of features for line

def parseCorpse(line):
    A=[(ingredientsIds.value[el],1) for el in line[1]]
    return A


# Extracting corpus

corpse=rawRDD.map(parseCorpse).collect()


# creating corpora dictionary

dictionary=corpora.Dictionary(rawRDD.map(lambda x:x[1]).collect())


# creating tf-idf model using gensim

model=models.TfidfModel(corpse)
tf_idf=model[corpse]


# lsi model using gensim

lsimodel=models.LsiModel(tf_idf,id2word=dictionary,num_topics=1000)


# converting cuisines-features matrix to lower ranked

docVectors=lsimodel[tf_idf]


# defining index for predictions

index = similarities.MatrixSimilarity(docVectors)


# creating list of cuisines for each row

cuisinesToPredict=rawRDD.map(lambda x:x[0]).collect()


# prediction function

def predict(line,numTop):
    doc=line[1]
    query=[(ingredientsIds.value[el],1) for el in doc if ingredientsIds.value.get(el,-1)>=0]
    query=model[query]
    query=lsimodel[query]
    inds=index[query]
    sims=list(enumerate(inds))
    sims=sorted(sims,key=lambda x:-x[1])
    A={}
    for i in range(numTop):
        A[cuisinesToPredict[sims[i][0]]]=A.get(cuisinesToPredict[sims[i][0]],0)+sims[i][1]
    A=A.items()
    A=sorted(A,key=lambda x:-x[1])
    return A[0][0]     


# parsing test data

tst=open('test.json','r')
string=tst.read().strip()
testJs=json.loads(string)
testList=[]
for item in testJs:
    testList.append((item['id'],item['ingredients']))
testSet=sc.parallelize(testList)


# obtaining predictions for test set

predictions=testSet.map(lambda x:(x[0],predict(x,20)))
predictions.saveAsTextFile('duuuupa')
