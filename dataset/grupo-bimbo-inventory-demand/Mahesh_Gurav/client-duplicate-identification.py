import numpy as np
import pandas as pd

client_info = pd.read_csv("../input/cliente_tabla.csv")

#print(client_info)

data_to_check = client_info.as_matrix(columns=['Cliente_ID'])
data_to_check = np.reshape(data_to_check, len(data_to_check))
counts = np.bincount(data_to_check)
duplicates = np.where(counts > 1)[0]
duplicates_length = len(duplicates)
print("-------------------------------------")
print("Number of duplicate client entries are : " + str(duplicates_length))
print("-------------------------------------")

#for index in range(0, duplicates_length):
#    print duplicates[index]
