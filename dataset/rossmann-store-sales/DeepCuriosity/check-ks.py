import numpy 

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

my_data = numpy.genfromtxt(train_file, delimiter='\t');

print(my_data.shape);

