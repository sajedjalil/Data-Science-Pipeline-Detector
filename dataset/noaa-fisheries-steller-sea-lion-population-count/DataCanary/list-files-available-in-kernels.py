# Because this dataset is larger than normal, kernels will only use a subset of the training data.
# Use kernels to demo your data munging and sketch out ideas...

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/Train"]).decode("utf8"))
print(check_output(["ls", "../input/TrainDotted"]).decode("utf8"))

