import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# Import the linearregression model.
from sklearn.linear_model import LinearRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
print ('Reading train!')
train = pd.read_csv('../input/train.csv',
                    usecols=['Agencia_ID',
                                  'Ruta_SAK',
                                  'Cliente_ID',
                                  'Producto_ID',
                                  'Demanda_uni_equil'],
                    dtype={'Agencia_ID': 'uint16',
                                      'Ruta_SAK': 'uint16',
                                      'Cliente_ID': 'int32',
                                      'Producto_ID': 'uint16',
                                      'Demanda_uni_equil': 'float32'})
# Get all the columns from the dataframe.
columns = train.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["Ruta_SAK", "Producto_ID", "Demanda_uni_equil"]]

# Store the variable we'll be predicting on.
target = "Demanda_uni_equil"


# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

print ('Reading Test')
test = pd.read_csv('../input/test.csv',
                   usecols=['Agencia_ID',
                              'Ruta_SAK',
                              'Cliente_ID',
                              'Producto_ID',
                            'id'],
                   dtype={'Agencia_ID': 'uint16',
                                      'Ruta_SAK': 'uint16',
                                      'Cliente_ID': 'int32',
                                      'Producto_ID': 'uint16',
                                      'id': 'int32'})
print ('Test read!')
# Generate our predictions for the test set.
predictions = model.predict(test[columns])
answers = predictions.tolist()
import csv
with open('answer.csv', 'w') as csvfile:
	reader = csv.DictReader(csvfile)
	spamwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	count = len(answers) - 1
	num = 1
	spamwriter.writerow(['id', 'Demanda_uni_equil'])
	while (count > 0):
	    value = answers[num]
	    spamwriter.writerow([num,value])
	    #csvfile.write(str(num)+","+predict_data[num]+"\n");
	    #spamwriter.write
	    count = count - 1
	    num = num + 1	