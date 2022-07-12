import numpy as np
import os

# os.system("ls ../input")

training_dataset = np.genfromtxt("../input/train.csv", delimiter=',')
print(training_dataset.shape)
output_file = "output.csv"
f = open(output_file, 'w')
f.flush()
f.write("id")