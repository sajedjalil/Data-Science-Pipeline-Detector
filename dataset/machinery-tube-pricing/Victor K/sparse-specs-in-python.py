import os
import pandas as pd
import numpy as np

def get_sparse_specs():
	input_folder = '../input/'
	specs = pd.read_csv(os.path.join(input_folder, 'specs.csv'))
	specs = specs.fillna('None')

	key_name = 'tube_assembly_id'
	keys = specs[key_name]
	specs_ = specs.drop([key_name], axis=1)
	all_values = sorted(pd.Series(specs_.values.ravel()).unique())

	groups = {}
	for i in range(0, len(specs)):
		group = specs.loc[i][1:].unique()
		groups[specs.loc[i][0]] = group

	rows = []
	for key in keys:
		row = [key] + list(map(lambda x: 1 if x in groups[key] else 0, all_values))
		rows.append(row)

	ret = pd.DataFrame(rows, columns = [key_name] + list(all_values))
	ret = ret.drop(['None'], axis=1)
	return ret
	
print(get_sparse_specs())