from pybrain.datasets import SupervisedDataSet

dataModel = [
    [(0,0), (0,)],
    [(0,1), (1,)],
    [(1,0), (1,)],
    [(1,1), (0,)],
]

ds = SupervisedDataSet(2, 1)
for input, target in dataModel:
    ds.addSample(input, target)

# create a large random data set
import random
random.seed()
trainingSet = SupervisedDataSet(2, 1);
for ri in range(0,1000):
    input,target = dataModel[random.getrandbits(2)];
    trainingSet.addSample(input, target)

from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 2, 1, bias=True)

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.00)
trainer.trainUntilConvergence(verbose=True,
                              trainingData=trainingSet,
                              validationData=ds,
                              maxEpochs=100)

print ('0,0->', net.activate([0,0]))
print ('0,1->', net.activate([0,1]))
print ('1,0->', net.activate([1,0]))
print ('1,1->', net.activate([1,1]))