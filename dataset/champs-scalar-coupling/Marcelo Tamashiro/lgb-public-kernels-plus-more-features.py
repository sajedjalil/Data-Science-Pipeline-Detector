# used features from:
# Giba's kernel https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb
# Cao's kernels https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28 
#               https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
# nosound's kernel https://www.kaggle.com/zaharch/quantum-machine-9-qm9

#things i've done:
#I make some features separating bonds and not bonds
#used randomforestclassifier prediction as input
#feature selection with lgb model for each type

#things that can be done to easily improve score in LB:
#cross validation / increase iterations 

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier
from scipy.stats import kurtosis, skew
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
    
    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_classes)
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        X = X.replace([np.inf,-np.inf], np.nan)
        X = X.fillna(0)
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            X = np.array(X)
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        X = X.replace([np.inf,-np.inf], np.nan)
        X = X.fillna(0)
        X = np.array(X)
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

all_features = ['type',  'atom_x', 'x_x', 'y_x','z_x', 'n_bonds_x', 'atom_y', 'x_y', 'y_y',
       'z_y', 'n_bonds_y', 'C', 'F', 'H', 'N', 'O', 'distance', 'dist_mean_x','dist_mean_y',
       'x_dist', 'y_dist', 'z_dist', 'x_dist_abs', 'y_dist_abs', 'z_dist_abs','inv_distance3']
cat_features = ['type','atom_x','atom_y']

class MoreStructureProperties(TransformerMixin, BaseEstimator):
    
    def __init__(self,atomic_radius,electronegativity):
        self.atomic_radius = atomic_radius
        self.electronegativity = electronegativity
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        atom_rad = [self.atomic_radius[x] for x in X['atom'].values]
        X['rad'] = atom_rad
        position = X[['x','y','z']].values
        p_temp = position
        molec_name = X['molecule_name'].values
        m_temp = molec_name
        radius = X['rad'].values
        r_temp = radius
        bond = 0
        dist_keep = 0
        dist_bond = 0 
        no_bond = 0
        dist_no_bond = 0
        dist_matrix = np.zeros((X.shape[0],2*29))
        dist_matrix_bond = np.zeros((X.shape[0],2*29))
        dist_matrix_no_bond = np.zeros((X.shape[0],2*29))
        
        for i in range(29):
            p_temp = np.roll(p_temp,-1,axis=0)
            m_temp = np.roll(m_temp,-1,axis=0)
            r_temp = np.roll(r_temp,-1,axis=0)
            mask = (m_temp==molec_name)
            dist = np.linalg.norm(position-p_temp,axis=1) * mask            
            dist_temp = np.roll(np.linalg.norm(position-p_temp,axis=1)*mask,i+1,axis=0)
            diff_radius_dist = (dist-(radius+r_temp)) * (dist<(radius+r_temp)) * mask
            diff_radius_dist_temp = np.roll(diff_radius_dist,i+1,axis=0)
            bond += (dist<(radius+r_temp)) * mask
            bond_temp = np.roll((dist<(radius+r_temp)) * mask,i+1,axis=0)
            no_bond += (dist>=(radius+r_temp)) * mask
            no_bond_temp = np.roll((dist>=(radius+r_temp)) * mask,i+1,axis=0)
            bond += bond_temp
            no_bond += no_bond_temp
            dist_keep += dist * mask
            dist_matrix[:,2*i] = dist
            dist_matrix[:,2*i+1] = dist_temp
            dist_matrix_bond[:,2*i] = dist * (dist<(radius+r_temp)) * mask
            dist_matrix_bond[:,2*i+1] = dist_temp * bond_temp
            dist_matrix_no_bond[:,2*i] = dist * (dist>(radius+r_temp)) * mask
            dist_matrix_no_bond[:,2*i+1] = dist_temp * no_bond_temp
        X['n_bonds'] = bond
        X['n_no_bonds'] = no_bond
        X['dist_mean'] = np.nanmean(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_median'] = np.nanmedian(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_std_bond'] = np.nanstd(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_mean_bond'] = np.nanmean(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_median_bond'] = np.nanmedian(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_mean_no_bond'] = np.nanmean(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_std_no_bond'] = np.nanstd(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_median_no_bond'] = np.nanmedian(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_std'] = np.nanstd(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_min'] = np.nanmin(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_max'] = np.nanmax(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['range_dist'] = np.absolute(X['dist_max']-X['dist_min'])
        X['dist_bond_min'] = np.nanmin(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_bond_max'] = np.nanmax(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['range_dist_bond'] = np.absolute(X['dist_bond_max']-X['dist_bond_min'])
        X['dist_no_bond_min'] = np.nanmin(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_no_bond_max'] = np.nanmax(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['range_dist_no_bond'] = np.absolute(X['dist_no_bond_max']-X['dist_no_bond_min'])
        X['n_diff'] = pd.DataFrame(np.around(dist_matrix_bond,5)).nunique(axis=1).values  #5
        X = reduce_mem_usage(X,verbose=False)
        return X
        
    
class MakeMoreFeatures(TransformerMixin, BaseEstimator):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['distance'] = np.linalg.norm(X[['x_x','y_x','z_x']].values - X[['x_y','y_y','z_y']].values ,axis=1)
        X['x_dist'] = X['x_x'] - X['x_y']
        X['y_dist'] = X['y_x'] - X['y_y']
        X['z_dist'] = X['z_x'] - X['z_y']
        X['x_dist_abs'] = np.absolute(X['x_dist'])
        X['y_dist_abs'] = np.absolute(X['y_dist'])
        X['z_dist_abs'] = np.absolute(X['z_dist'])
        X['inv_distance3'] = 1/(X['distance']**3)
        X['dimension_x'] = np.absolute(X.groupby(['molecule_name'])['x_x'].transform('max') - X.groupby(['molecule_name'])['x_x'].transform('min'))
        X['dimension_y'] = np.absolute(X.groupby(['molecule_name'])['y_x'].transform('max') - X.groupby(['molecule_name'])['y_x'].transform('min'))
        X['dimension_z'] = np.absolute(X.groupby(['molecule_name'])['z_x'].transform('max') - X.groupby(['molecule_name'])['z_x'].transform('min'))
        X['molecule_dist_mean_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('mean')
        X['molecule_dist_mean_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('mean')
        X['molecule_dist_mean_bond_x'] = X.groupby(['molecule_name'])['dist_mean_bond_x'].transform('mean')
        X['molecule_dist_mean_bond_y'] = X.groupby(['molecule_name'])['dist_mean_bond_y'].transform('mean')
        X['molecule_dist_range_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('max') - X.groupby(['molecule_name'])['dist_mean_x'].transform('min')
        X['molecule_dist_range_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('max') - X.groupby(['molecule_name'])['dist_mean_y'].transform('min')
        X['molecule_dist_std_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('std')
        X['molecule_dist_std_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('std')
        X['molecule_atom_0_dist_mean'] = X.groupby(['molecule_name','atom_x'])['distance'].transform('mean')
        X['molecule_atom_1_dist_mean'] = X.groupby(['molecule_name','atom_y'])['distance'].transform('mean')
        X['molecule_atom_0_dist_std_diff'] = X.groupby(['molecule_name', 'atom_x'])['distance'].transform('std') - X['distance']
        X['molecule_atom_1_dist_std_diff'] = X.groupby(['molecule_name', 'atom_y'])['distance'].transform('std') - X['distance']
        X['molecule_type_dist_min'] = X.groupby(['molecule_name','type'])['distance'].transform('min') 
        X['molecule_type_dist_max'] = X.groupby(['molecule_name','type'])['distance'].transform('max') 
        X['molecule_dist_mean_no_bond_x'] = X.groupby(['molecule_name'])['dist_mean_no_bond_x'].transform('mean')
        X['molecule_dist_mean_no_bond_y'] = X.groupby(['molecule_name'])['dist_mean_no_bond_y'].transform('mean')
        X['molecule_atom_index_0_dist_min'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min') #new variable - dont include
        X['molecule_atom_index_0_dist_std'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('std') #new variable - dont include
        X['molecule_atom_index_0_dist_min_div'] = X['molecule_atom_index_0_dist_min']/X['distance'] #new variable - include
        X['molecule_atom_index_0_dist_std_div'] = X['molecule_atom_index_0_dist_std']/X['distance'] #new variable - include
        X['molecule_atom_index_0_dist_mean'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('mean') #new variable - include
        X['molecule_atom_index_0_dist_max'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max') #new variable - include
        X['molecule_atom_index_0_dist_mean_diff'] = X['molecule_atom_index_0_dist_mean'] - X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_mean'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('mean') #new variable - include
        X['molecule_atom_index_1_dist_max'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('max') #new variable - include
        X['molecule_atom_index_1_dist_min'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('min') #new variable - include
        X['molecule_atom_index_1_dist_std'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('std') #new variable - dont include
        X['molecule_atom_index_1_dist_min_div'] = X['molecule_atom_index_1_dist_min']/X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_std_diff'] = X['molecule_atom_index_1_dist_std'] - X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_mean_div'] = X['molecule_atom_index_1_dist_mean']/X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_min_diff'] = X['molecule_atom_index_1_dist_min_div'] - X['distance'] #new variable - include
        le = LabelEncoder()
        for feat in ['atom_x','atom_y']:
            le.fit(X[feat])
            X[feat] = le.transform(X[feat])
        X = reduce_mem_usage(X,verbose=False)
        return X
    

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import gc
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

t0 = time()


def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    return df

    
def find_dist(df):
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values
    
    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
    df['dist_inv2'] = 1/df['dist']**2
    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2
    return df

def find_closest_atom(df):    
    df_temp = df.loc[:,["molecule_name",
                      "atom_index_0","atom_index_1",
                      "dist","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                       'atom_index_1': 'atom_index_0',
                                       'x_0': 'x_1',
                                       'y_0': 'y_1',
                                       'z_0': 'z_1',
                                       'x_1': 'x_0',
                                       'y_1': 'y_0',
                                       'z_1': 'z_0'})
    df_temp_all = pd.concat((df_temp,df_temp_),axis=0)

    df_temp_all["min_distance"]=df_temp_all.groupby(['molecule_name', 
                                                     'atom_index_0'])['dist'].transform('min')
    df_temp_all["max_distance"]=df_temp_all.groupby(['molecule_name', 
                                                     'atom_index_0'])['dist'].transform('max')
    
    df_temp = df_temp_all[df_temp_all["min_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_closest',
                                         'dist': 'distance_closest',
                                         'x_1': 'x_closest',
                                         'y_1': 'y_closest',
                                         'z_1': 'z_closest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])
    
    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                        'distance_closest': f'distance_closest_{atom_idx}',
                                        'x_closest': f'x_closest_{atom_idx}',
                                        'y_closest': f'y_closest_{atom_idx}',
                                        'z_closest': f'z_closest_{atom_idx}'})
        
    df_temp= df_temp_all[df_temp_all["max_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','max_distance'], axis=1)
    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_farthest',
                                         'dist': 'distance_farthest',
                                         'x_1': 'x_farthest',
                                         'y_1': 'y_farthest',
                                         'z_1': 'z_farthest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])
        
    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',
                                        'distance_farthest': f'distance_farthest_{atom_idx}',
                                        'x_farthest': f'x_farthest_{atom_idx}',
                                        'y_farthest': f'y_farthest_{atom_idx}',
                                        'z_farthest': f'z_farthest_{atom_idx}'})
    return df


def add_cos_features(df):
    
    df["distance_center0"] = np.sqrt((df['x_0']-df['c_x'])**2 \
                                   + (df['y_0']-df['c_y'])**2 \
                                   + (df['z_0']-df['c_z'])**2)
    df["distance_center1"] = np.sqrt((df['x_1']-df['c_x'])**2 \
                                   + (df['y_1']-df['c_y'])**2 \
                                   + (df['z_1']-df['c_z'])**2)
    
    df['distance_c0'] = np.sqrt((df['x_0']-df['x_closest_0'])**2 + \
                                (df['y_0']-df['y_closest_0'])**2 + \
                                (df['z_0']-df['z_closest_0'])**2)
    df['distance_c1'] = np.sqrt((df['x_1']-df['x_closest_1'])**2 + \
                                (df['y_1']-df['y_closest_1'])**2 + \
                                (df['z_1']-df['z_closest_1'])**2)
    
    df["distance_f0"] = np.sqrt((df['x_0']-df['x_farthest_0'])**2 + \
                                (df['y_0']-df['y_farthest_0'])**2 + \
                                (df['z_0']-df['z_farthest_0'])**2)
    df["distance_f1"] = np.sqrt((df['x_1']-df['x_farthest_1'])**2 + \
                                (df['y_1']-df['y_farthest_1'])**2 + \
                                (df['z_1']-df['z_farthest_1'])**2)
    
    vec_center0_x = (df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)
    vec_center0_y = (df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)
    vec_center0_z = (df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)
    
    vec_center1_x = (df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)
    vec_center1_y = (df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)
    vec_center1_z = (df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)
    
    vec_c0_x = (df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_y = (df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_z = (df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)
    
    vec_c1_x = (df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_y = (df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_z = (df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)
    
    vec_f0_x = (df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_y = (df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_z = (df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)
    
    vec_f1_x = (df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_y = (df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_z = (df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)
    
    vec_x = (df['x_1']-df['x_0'])/df['dist']
    vec_y = (df['y_1']-df['y_0'])/df['dist']
    vec_z = (df['z_1']-df['z_0'])/df['dist']
    
    df["cos_c0_c1"] = vec_c0_x*vec_c1_x + vec_c0_y*vec_c1_y + vec_c0_z*vec_c1_z
    df["cos_f0_f1"] = vec_f0_x*vec_f1_x + vec_f0_y*vec_f1_y + vec_f0_z*vec_f1_z
    
    df["cos_c0_f0"] = vec_c0_x*vec_f0_x + vec_c0_y*vec_f0_y + vec_c0_z*vec_f0_z
    df["cos_c1_f1"] = vec_c1_x*vec_f1_x + vec_c1_y*vec_f1_y + vec_c1_z*vec_f1_z
    
    df["cos_center0_center1"] = vec_center0_x*vec_center1_x \
                              + vec_center0_y*vec_center1_y \
                              + vec_center0_z*vec_center1_z
    
    df["cos_c0"] = vec_c0_x*vec_x + vec_c0_y*vec_y + vec_c0_z*vec_z
    df["cos_c1"] = vec_c1_x*vec_x + vec_c1_y*vec_y + vec_c1_z*vec_z
    
    df["cos_f0"] = vec_f0_x*vec_x + vec_f0_y*vec_y + vec_f0_z*vec_z
    df["cos_f1"] = vec_f1_x*vec_x + vec_f1_y*vec_y + vec_f1_z*vec_z
    
    df["cos_center0"] = vec_center0_x*vec_x + vec_center0_y*vec_y + vec_center0_z*vec_z
    df["cos_center1"] = vec_center1_x*vec_x + vec_center1_y*vec_y + vec_center1_z*vec_z

    return df

def dummies(df, list_cols):
    for col in list_cols:
        df_dummies = pd.get_dummies(df[col], drop_first=True, 
                                    prefix=(str(col)))
        df = pd.concat([df, df_dummies], axis=1)
    return df


def add_qm9_features(df):
    data_qm9 = pd.read_pickle('../input/quantum-machine-9-qm9/data.covs.pickle')
    to_drop = ['type', 
               'linear', 
               'atom_index_0', 
               'atom_index_1', 
               'scalar_coupling_constant', 
               'U', 'G', 'H', 
               'mulliken_mean', 'r2', 'U0']
    data_qm9 = data_qm9.drop(columns = to_drop, axis=1)
    data_qm9 = reduce_mem_usage(data_qm9,verbose=False)
    df = pd.merge(df, data_qm9, how='left', on=['molecule_name','id'])
    del data_qm9
    
    df = dummies(df, ['type', 'atom_1'])
    return df

def get_features(df, struct):
    for atom_idx in [0,1]:
        df = map_atom_info(df, struct, atom_idx)
        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
        struct['c_x'] = struct.groupby('molecule_name')['x'].transform('mean')
        struct['c_y'] = struct.groupby('molecule_name')['y'].transform('mean')
        struct['c_z'] = struct.groupby('molecule_name')['z'].transform('mean')

    df = find_dist(df)
    df = find_closest_atom(df)
    df = add_cos_features(df)
    df = add_qm9_features(df)
    return df

def comp_score (y_true, y_pred, jtype):
    df = pd.DataFrame()
    df['y_true'] , df['y_pred'], df['jtype'] = y_true , y_pred, jtype
    score = 0 
    for t in jtype.unique():
        score_jtype = np.log(mean_absolute_error(df[df.jtype==t]['y_true'],df[df.jtype==t]['y_pred']))
        score += score_jtype
        print(f'{t} : {score_jtype}')
    score /= len(jtype.unique())
    return score

def feat_from_structures(df, st):
    df = pd.merge(df,st,how='left',left_on=['molecule_name','atom_index_0'], right_on=['molecule_name','atom_index'])
    df = pd.merge(df,st,how='left',left_on=['molecule_name','atom_index_1'], right_on=['molecule_name','atom_index'])
    n_atoms = st.groupby(['molecule_name','atom'])['atom'].size().to_frame(name = 'count').reset_index()
    n_atoms_df = n_atoms.pivot_table('count',['molecule_name'], 'atom')
    n_atoms_df.fillna(0,inplace=True)
    df = pd.merge(df,n_atoms_df,on=['molecule_name'],how='left')
    del n_atoms
    gc.collect()
    return df

atomic_radius = {'H': 0.43, 'C': 0.82, 'N': 0.8, 'O': 0.78, 'F': 0.76}
electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}


struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
pipeline_model1 = make_pipeline(MoreStructureProperties(atomic_radius,electronegativity))
pipeline_model2 = make_pipeline(MakeMoreFeatures())
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
structures_yukawa = pd.read_csv('../input/parallelization-of-coulomb-yukawa-interaction/structures_yukawa.csv')
struct = pd.concat([struct, structures_yukawa], axis=1)
del structures_yukawa
struct = reduce_mem_usage(struct,verbose=False)
gc.collect()
train = get_features(train, struct.copy())
test = get_features(test, struct.copy())
y = train['scalar_coupling_constant']
del struct
gc.collect()

struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
struct = pipeline_model1.fit_transform(struct)
train = feat_from_structures(train,struct)
train = pipeline_model2.fit_transform(train.drop(['scalar_coupling_constant'],axis=1), train['scalar_coupling_constant'])
test = feat_from_structures(test,struct)
test = pipeline_model2.transform(test)
train = reduce_mem_usage(train,verbose=False)
test = reduce_mem_usage(test,verbose=False)

giba_columns = ['inv_dist0', 'inv_dist1', 'inv_distP', 'inv_dist0R', 'inv_dist1R', 'inv_distPR', 'inv_dist0E', 'inv_dist1E', 'inv_distPE', 'linkM0',
         'linkM1', 'min_molecule_atom_0_dist_xyz', 'mean_molecule_atom_0_dist_xyz', 'max_molecule_atom_0_dist_xyz', 'sd_molecule_atom_0_dist_xyz', 'min_molecule_atom_1_dist_xyz',
         'mean_molecule_atom_1_dist_xyz', 'max_molecule_atom_1_dist_xyz', 'sd_molecule_atom_1_dist_xyz', 'coulomb_C.x', 'coulomb_F.x', 'coulomb_H.x', 'coulomb_N.x',
         'coulomb_O.x', 'yukawa_C.x', 'yukawa_F.x', 'yukawa_H.x', 'yukawa_N.x', 'yukawa_O.x', 'vander_C.x', 'vander_F.x', 'vander_H.x', 'vander_N.x', 'vander_O.x',
         'coulomb_C.y', 'coulomb_F.y', 'coulomb_H.y', 'coulomb_N.y', 'coulomb_O.y', 'yukawa_C.y', 'yukawa_F.y', 'yukawa_H.y', 'yukawa_N.y', 'yukawa_O.y', 'vander_C.y',
         'vander_F.y', 'vander_H.y', 'vander_N.y', 'vander_O.y', 'distC0', 'distH0', 'distN0', 'distC1', 'distH1', 'distN1', 'adH1', 'adH2', 'adH3', 'adH4', 'adC1',
         'adC2', 'adC3', 'adC4', 'adN1', 'adN2', 'adN3', 'adN4', 'NC', 'NH', 'NN', 'NF', 'NO']

train_giba_t = pd.read_csv('../input/giba-molecular-features/train_giba.csv/train_giba.csv',
                        header=0,  usecols=giba_columns)
test_giba_t = pd.read_csv('../input/giba-molecular-features/test_giba.csv/test_giba.csv',
                       header=0,  usecols=giba_columns)
train_giba_t = reduce_mem_usage(train_giba_t, verbose=False)
test_giba_t = reduce_mem_usage(test_giba_t, verbose=False)

train = pd.concat((train,train_giba_t),axis=1)
test = pd.concat((test,test_giba_t),axis=1)

all_features = ['type',   'x_x', 'y_x','z_x', 'atom_y', 'x_y', 'y_y',
       'z_y', 'n_bonds_y', 'C', 'F', 'H', 'N', 'O', 'distance', 'dist_mean_x','dist_mean_y',
        'x_dist_abs', 'y_dist_abs', 'z_dist_abs','inv_distance3',
       'molecule_atom_1_dist_std_diff','molecule_dist_mean_x',
       'molecule_dist_mean_y','molecule_dist_std_x','molecule_dist_std_y','molecule_atom_0_dist_mean',
       'molecule_atom_1_dist_mean','dist_mean_bond_y',
       'n_no_bonds_x','n_no_bonds_y', 'dist_std_x', 'dist_std_y','dist_min_x','dist_min_y','dist_max_x', 'dist_max_y',
       'molecule_dist_range_x','molecule_dist_range_y', 'dimension_x', 'dimension_y','dimension_z','molecule_dist_mean_bond_x',
       'molecule_dist_mean_bond_x','dist_mean_no_bond_x','dist_mean_no_bond_y',
       'dist_std_bond_y','dist_bond_min_y','dist_bond_max_y',
       'range_dist_bond_y','dist_std_no_bond_x','dist_std_no_bond_y', 'dist_no_bond_min_x','dist_no_bond_min_y','dist_no_bond_max_x',
       'dist_no_bond_max_y', 'range_dist_no_bond_x','range_dist_no_bond_y','dist_median_bond_y','dist_median_x',
       'dist_median_y','dist_median_no_bond_x','dist_median_no_bond_y','molecule_type_dist_min','molecule_type_dist_max',
       'molecule_dist_mean_no_bond_x','molecule_dist_mean_no_bond_y', 'n_diff_y','molecule_atom_index_0_dist_min_div','molecule_atom_index_0_dist_std_div',
        'molecule_atom_index_0_dist_mean','molecule_atom_index_0_dist_max','molecule_atom_index_1_dist_mean','molecule_atom_index_1_dist_max',
       'molecule_atom_index_1_dist_min','molecule_atom_index_1_dist_min_div','molecule_atom_index_1_dist_std_diff','molecule_atom_index_0_dist_mean_diff',
        'molecule_atom_index_1_dist_mean_div','molecule_atom_index_1_dist_min_diff', 'rc_A', 'rc_B', 'rc_C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'zpve', 'Cv',
         'freqs_min', 'freqs_max', 'freqs_mean', 'mulliken_min', 'mulliken_max', 'mulliken_atom_0', 'mulliken_atom_1',
         'dist_C_0_x', 'dist_C_1_x', 'dist_C_2_x', 'dist_C_3_x', 'dist_C_4_x', 'dist_F_0_x', 'dist_F_1_x', 'dist_F_2_x', 'dist_H_0_x',
         'dist_H_1_x', 'dist_H_2_x', 'dist_H_3_x', 'dist_H_4_x', 'dist_N_0_x', 'dist_N_1_x', 'dist_N_2_x', 'dist_N_3_x', 'dist_N_4_x', 'dist_O_0_x', 'dist_O_1_x',
         'dist_O_2_x', 'dist_O_3_x', 'dist_O_4_x', 'dist_C_0_y', 'dist_C_1_y', 'dist_C_2_y', 'dist_C_3_y', 'dist_C_4_y', 'dist_F_0_y', 'dist_F_1_y', 'dist_F_2_y',
         'dist_F_3_y', 'dist_F_4_y', 'dist_H_0_y', 'dist_H_1_y', 'dist_H_2_y', 'dist_H_3_y', 'dist_H_4_y', 'dist_N_0_y', 'dist_N_1_y', 'dist_N_2_y', 'dist_N_3_y',
         'dist_N_4_y', 'dist_O_0_y', 'dist_O_1_y', 'dist_O_2_y', 'dist_O_3_y', 'dist_O_4_y','distance_closest_0', 'distance_closest_1', 'distance_farthest_0',
         'distance_farthest_1','cos_c0_c1', 'cos_f0_f1','cos_c0_f0', 'cos_c1_f1', 'cos_center0_center1', 'cos_c0', 'cos_c1', 'cos_f0', 'cos_f1',
         'cos_center0', 'cos_center1'] + giba_columns

cat_features = ['atom_y']

X = train[all_features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2 , random_state=182)
pred = np.zeros(X_val.shape[0])

params = {'num_leaves': 50,
          'min_child_samples': 79,
          'min_data_in_leaf': 100,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0,
          'num_iterations': 4000
         }
d_map_val = dict(zip(list(y_val.index.values),list(np.arange(y_val.shape[0]))))


X_test = test[all_features]
sub = pd.DataFrame()
sub['id'] = test['id']
sub['type'] = test['type']
pred_sub = np.zeros(sub.shape[0])
all_features.pop(0)
unique_types = train['type'].unique()
del train
del test
del struct
del X
del y
del train_giba_t
del test_giba_t
gc.collect()

gc.collect()
rf_cols = ['rf_00','rf_01','rf_02','rf_03','rf04','rf05']
rf_cols1 = ['rf_10','rf_11','rf12']
for t in unique_types:
    evals_result = {}
    idx_train = X_train[X_train.type==t].index.values
    idx_val = X_val[X_val.type==t].index.values
    idx_sub = sub[sub.type==t].index.values
    clf = ClassifierTransformer(RandomForestClassifier(),n_classes=5,cv=5)
    clf1 = ClassifierTransformer(RandomForestClassifier(),n_classes=2,cv=5)
    X_extra = np.hstack([clf.fit_transform(X_train[X_train.type==t].drop(['type'],axis=1), y_train[idx_train]),clf1.fit_transform(X_train[X_train.type==t].drop(['type'],axis=1), y_train[idx_train])])
    X_extra_val = np.hstack([clf.transform(X_val[X_val.type==t].drop(['type'],axis=1)),clf1.transform(X_val[X_val.type==t].drop(['type'],axis=1))])
    X_extra_test = np.hstack([clf.transform(X_test[X_test.type==t].drop(['type'],axis=1)),clf1.transform(X_test[X_test.type==t].drop(['type'],axis=1))])
    X_p = pd.DataFrame(data=np.hstack([X_train[X_train.type==t].drop(['type'],axis=1).values,X_extra]), columns=list(X_train.drop(['type'],axis=1).columns)+rf_cols+rf_cols1)
    X_p_val = pd.DataFrame(data=np.hstack([X_val[X_val.type==t].drop(['type'],axis=1).values,X_extra_val]), columns=list(X_val.drop(['type'],axis=1).columns)+rf_cols+rf_cols1)
    X_p_test = pd.DataFrame(data=np.hstack([X_test[X_test.type==t].drop(['type'],axis=1).values,X_extra_test]), columns=list(X_test.drop(['type'],axis=1).columns)+rf_cols+rf_cols1)
    gbm = lgb.LGBMRegressor()
    gbm.fit(X_p, y_train[idx_train])
    gbm.booster_.feature_importance()
    
    fea_imp_ = pd.DataFrame({'cols':X_p.columns, 'fea_imp':gbm.feature_importances_})
    if t=='1JHC':
        nb_feat=100
    else:
        nb_feat=90
    remain_features = list(fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)['cols'].values[:nb_feat])
    
    
    print(f'Training for {t}')
    lgb_train = lgb.Dataset(X_p[remain_features],label=y_train[idx_train])
    lgb_val = lgb.Dataset(X_p_val[remain_features],label=y_val[idx_val])
    model = lgb.train(params=params, train_set=lgb_train,  valid_sets=[lgb_train,lgb_val], valid_names=['train','val'],
                  fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto',evals_result=evals_result,
                  early_stopping_rounds=50,  verbose_eval=500)
    val_map = [d_map_val[k] for k in list(idx_val)]
    pred[val_map] = model.predict(X_p_val[remain_features])
    pred_sub[idx_sub] = model.predict(X_p_test[remain_features])
    del X_p
    del X_p_val
    del X_p_test
    del X_extra
    del X_extra_val
    del X_extra_test
    del clf
    del clf1
    del gbm
    gc.collect()

    ax = lgb.plot_metric(evals_result, metric='l1')
    plt.title(f'{t}_mae')
    plt.show()
    plt.savefig(f'{t}_mae.png')
    plt.clf()
    ax = lgb.plot_importance(model, max_num_features=40)
    plt.title(f'{t}_feature_importance')
    plt.show()
    plt.savefig(f'{t}_feature_importance.png')
    plt.clf()
  
    sns.scatterplot(x=pred[val_map],y=y_val[idx_val]-pred[val_map])
    plt.xlabel('target')
    plt.ylabel('residual')
    plt.title(f'{t}_residuals_plot')
    plt.show()
    plt.savefig(f'{t}_residuals_plot.png')
    plt.clf()
jtype = X_val['type']
print('Competition Validation Score: ',comp_score(y_val,pred,jtype))

print(f'Time to train: {time()-t0}')    
    
sub['scalar_coupling_constant'] = pred_sub
sub[['id','scalar_coupling_constant']].to_csv('sub_lgb_model_individual.csv',index=False)
