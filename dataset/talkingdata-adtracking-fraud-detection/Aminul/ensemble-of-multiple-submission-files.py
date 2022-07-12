#special thanks to Ryan Epp for his share of this code
import pandas as pd
import numpy as np
import os

# Controls weights when combining predictions
# 0: equal average of all inputs;
# 1: up to 50% of weight going to least correlated input
DENSITY_COEFF = 0.1
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

# When merging 2 files with corr > OVER_CORR_CUTOFF
# the result's weight is the max instead of the sum of the merged files' weights
OVER_CORR_CUTOFF = 0.99
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

#Put all the .csv file in ensemble folder
INPUT_DIR = '/home/shaikat/kaggle/m-rcnn/talking_data_fraud/ensemble/'

def load_submissions():
    files = os.listdir(INPUT_DIR)
    csv_files = []
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(f)
    frames = {f:pd.read_csv(INPUT_DIR+f).sort_values('click_id') for f in csv_files}
    return frames


def get_corr_mat(col,frames):
    c = pd.DataFrame()
    for name,df in frames.items():
        c[name] = df[col]
    cor = c.corr()
    for name in cor.columns:
        cor.set_value(name,name,0.0)
    return cor


def highest_corr(mat,frames):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr,f1,f2


def get_merge_weights(m1,m2,densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5*DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5*DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(col,frames,densities):
    if len(frames) == 1:
        _, fr = frames.popitem()
        return fr[col]

    mat = get_corr_mat(col,frames)
    corr,merge1,merge2 = highest_corr(mat,frames)
    new_col_name = merge1 + '_' + merge2

    w1,w2 = get_merge_weights(merge1,merge2,densities)
    new_df = pd.DataFrame()
    new_df[col] = (frames[merge1][col]*w1) + (frames[merge2][col]*w2)
    del frames[merge1]
    del frames[merge2]
    frames[new_col_name] = new_df

    if corr >= OVER_CORR_CUTOFF:
        print('\t',merge1,merge2,'  (OVER CORR)')
        densities[new_col_name] = max(densities[merge1],densities[merge2])
    else:
        print('\t',merge1,merge2)
        densities[new_col_name] = densities[merge1] + densities[merge2]

    del densities[merge1]
    del densities[merge2]
    #print(densities)
    return ensemble_col(col,frames,densities)


ens_submission = pd.read_csv('sample_submission.csv').sort_values('click_id')
#print(get_corr_mat('toxic',load_submissions()))

for col in ["is_attributed"]:
    frames = load_submissions()
    print('\n\n',col)
    densities = {k:1.0 for k in frames.keys()}
    ens_submission[col] = ensemble_col(col,frames,densities)

print(ens_submission)
ens_submission.to_csv('ensemble_submission.csv', index=False)