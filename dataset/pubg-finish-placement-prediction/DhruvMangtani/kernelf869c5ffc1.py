import requests
import json
import pandas as pd
import math

testFile = pd.read_csv("../input/test_V2.csv")

sample_ids = testFile["Id"]
X = testFile.iloc[:, 3:28].values

rows = X.shape[0]

def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def forwardProp(sample, hidden_layer_weights, output_layer_weights):
    hiddenLayer = []
    # connections to the hidden layer
    for neuron_weights in hidden_layer_weights:
        sum = 0.0
        for input in range(len(neuron_weights)):
            sum += float(sample[input]) * neuron_weights[input]
        hiddenLayer.append(sigmoid(sum))

    output = []
    sum = 0.0
    for output_weight in range(len(output_layer_weights)):
        sum += output_layer_weights[output_weight] * hiddenLayer[output_weight]

    return sigmoid(sum), hiddenLayer

def predictTestSamples():
    print("Training neural net online...")
    response = requests.get("https://sdnismcaei.execute-api.us-east-1.amazonaws.com/prod/train")
    jsonResponse = json.loads(response.text)

    hiddenLayerWeights = jsonResponse["hidden_layer"]
    outputLayerWeights = jsonResponse["output_layer"]

    print("Neural net trained")

    output_file = open("submission.csv", "w+")
    output_file.write("Id,winPlacePerc\n")
    for row in range(rows):
        sample = X[row].tolist()
        # delete the String input because we don't want to handle that
        del sample[12]
        sample[:] = [j / 2000 for j in sample]
        sample_output, _ = forwardProp(sample, hiddenLayerWeights, outputLayerWeights)
        output_file.write(sample_ids[row] + "," + str(sample_output) + "\n")

predictTestSamples()
