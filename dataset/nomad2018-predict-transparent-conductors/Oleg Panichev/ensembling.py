import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# List of files with submissions
files = [
    '../input/using-the-atomic-coordinates-for-prediction/sub.csv',
    '../input/hands-on-cubist/cub.csv',
    '../input/simple-catboost/0.0426663251101_.csv'
]

# Load submissions
p_buf = []
for file in files:
    df = pd.read_csv(file)
    df = df.sort_values('id')
    ids = df['id'].values
    p = df[['formation_energy_ev_natom', 'bandgap_energy_ev']].values
    p_buf.append(p)

# Generate predictions and save to file    
preds = np.mean(p_buf, axis=0)
subm = pd.DataFrame()
subm['id'] = ids
subm['formation_energy_ev_natom'] = preds[:, 0]
subm['bandgap_energy_ev'] = preds[:, 1]
subm.to_csv('ensemble_submission.csv', index=False)
