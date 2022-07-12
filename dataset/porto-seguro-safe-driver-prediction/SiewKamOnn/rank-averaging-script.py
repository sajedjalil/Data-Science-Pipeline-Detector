# inspired by Pavel's script at https://www.kaggle.com/pavetr/stacking-lb-0-285
# this script just requires typing the names of files to stack 
# at the initial section and it is ready to run

import pandas as pd
import numpy as np

# key in the files to stack here
# START
files = {}
files['file1'] = 'submission_file1.csv'
files['file2'] = 'submission_file2.csv'
files['file2'] = 'submission_file3.csv'
# END


dfs = []
for key, _file in files.items():
    df = pd.read_csv(_file)
    df.rename(columns = {'target': key}, inplace=True)
    dfs.append(df)



_submission = pd.concat(dfs, axis=1)


for key in files.keys():
    _submission[key + '_rank'] = _submission[key].rank()


_submission['rank_sum'] = np.sum(
        _submission[col] for col in _submission.columns if '_rank' in col)
_submission['target'] = _submission['rank_sum']/(len(files) *
        _submission.shape[0])

# take the first (id) and last column (target)
submission = _submission.iloc[:, [0, -1]]


filename = f"rank_average-{','.join(files.keys())}"
submit_file = f'{filename}.csv'
print(f'creating {submit_file}')

submission.to_csv(submit_file, index=False)
print('Done')
