RANDOM_STATE = 1337

import pandas as pd
import numpy as np
np.random.seed(RANDOM_STATE)

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#import ase
from ase.io import read
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.io import write
from ase import *
from math import *
from ase.calculators.abinit import Abinit
from ase.visualize import view


def rmsle(y_pred, y_target):
    """Computation of the root mean squared logarithmic error.
    
    Requirements:
        numpy
    
    Args:
        y_pred (numpy float array): The predicted values
        y_target (numpy float array): The target values
    
    Remark:
        The input arrays have to be of same length
    
    Returns:
        float: Root mean squared logarithmic error of the prediction.

    """
    assert len(y_pred) == len(y_target)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_target), 2)))
    
def save_submission(prediction_1, prediction_2, test_id):
    """The function saves the prediction values to a csv file
    
    Requirements:
        pandas
    
    Args:
        prediction_1 (numpy float array): The predicted values for the first target: formation_energy_ev_natom
        prediction_2 (numpy float array): The predicted values for the first target: bandgap_energy_ev
        test_id (numpy int array): The ids of the test data
    """

    submission = pd.concat([test_id, pd.DataFrame(prediction_1), pd.DataFrame(prediction_2)], axis=1)
    submission.columns = ['id','formation_energy_ev_natom', 'bandgap_energy_ev']
    submission.to_csv('submission.csv', index = False)
    
def get_xyz_data(filename, ids):
    """The function loads the xyz-geometry files and transforms it into a common python format (pandas table)
    
    Remark:
        The parts for read and split are adopted from Tony Y: https://www.kaggle.com/tonyyy
    
    Requirements:
        pandas
    
    Args:
        filename (string): path of the xyz file
        ids (integer): id of the corresponding entry in train data
        
    Returns:
        pandas dataframe A: Geometry data from the xyz file in table format
        pandas dataframe B: Lattice data from the xyz file in table format
        
    """
    
    A = pd.DataFrame(columns=list('ABCDE'))
    B = pd.DataFrame(columns=list('ABCE'))
    
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':

                newrowA = pd.DataFrame([[x[1],x[2],x[3],x[4],ids]], columns=list('ABCDE'))
                A = A.append(newrowA)
                
            elif x[0] == 'lattice_vector':
                
                newrowB = pd.DataFrame([[x[1],x[2],x[3],ids]], columns=list('ABCE'))
                B = B.append(newrowB)

    return A, B
    
def one_hot(df):
    """The function performs one hot encoding on the spacegroup column, which is not ordinal.

    Requirements:
        pandas
    
    Args:
        df (pandas dataframe): Data table with spacegroup column
    
    Returns:
        pandas dataframe df: Dataframe with one hot encoded spacegroup column
        
    """
    
    s = pd.Series(df["spacegroup"])
    t = pd.get_dummies(s)
    
    df["spacegroup_12"] = t[12]
    df["spacegroup_33"] = t[33]
    df["spacegroup_167"] = t[167]
    df["spacegroup_194"] = t[194]
    df["spacegroup_206"] = t[206]
    df["spacegroup_227"] = t[227]
    df = df.drop("spacegroup", axis = 1)
    
    return df
    
def feature_extraction(df, df1, n):
 
    """The function performs feature extraction on the information from xyz files
    
    Requirements:
        aes
        numpy
    
    Args:
        df (pandas dataframe): Data table with spatial distribution and atom info
        df1 (pandas dataframe): Data table with lattice vectors
        n (int): length of feature matrix for preallocation
        
    Returns:
        pandas dataframe feat_matrix: Table with features which are extracted from the input data
        
    """
    
    feat_matrix = pd.DataFrame(range(1, n + 1), columns=["id"])
    
    atoms = df
    lattices = df1

    mass_center_x = np.zeros(n, dtype=float)
    mass_center_y = np.zeros(n, dtype=float)
    mass_center_z = np.zeros(n, dtype=float)
    volume = np.zeros(n, dtype=float)
    min_distance = np.zeros(n, dtype=float)
    mean_distance = np.zeros(n, dtype=float)
    max_distance = np.zeros(n, dtype=float)
    variance_x = np.zeros(n, dtype=float)
    variance_y = np.zeros(n, dtype=float)
    variance_z = np.zeros(n, dtype=float)
    
    for index in range(n):
        
        lat = lattices[lattices["E"]==index+1]
        mol = atoms[atoms["E"]==index+1]
        atom_name = mol["D"].values
        lat = lat.drop(["Unnamed: 0", "E"], axis = 1)
        mol = mol.drop(["Unnamed: 0", "D", "E"], axis = 1)
        lat = lat.as_matrix()
        mol = mol.as_matrix()
        lst = []
        for k in range(0, len(mol)):
            lst.append(Atom(atom_name[k], (mol[k][0], mol[k][1], mol[k][2])))
        atom_conf = Atoms(lst)
        cell = [(lat[0][0], lat[0][1], lat[0][2]),
                (lat[1][0], lat[1][1], lat[1][2]),
                (lat[2][0], lat[2][1], lat[2][2])]
        atom_conf.set_cell(cell, scale_atoms=True)
        calc = Abinit()
        atom_conf.set_calculator(calc)
        
        mass_center_x[index] = atom_conf.get_center_of_mass()[0]
        mass_center_y[index] = atom_conf.get_center_of_mass()[1]
        mass_center_z[index] = atom_conf.get_center_of_mass()[2]
        volume[index] = atom_conf.get_volume()
        
        min_distance[index] = np.min(atom_conf.get_all_distances()+100*np.identity(len(atom_conf)))
        mean_distance[index] = np.mean(atom_conf.get_all_distances())
        max_distance[index] = np.max(atom_conf.get_all_distances())
        
        matrix = atom_conf.get_scaled_positions()
        variance_x[index] = np.var(matrix[:,0])
        variance_y[index] = np.var(matrix[:,1])
        variance_z[index] = np.var(matrix[:,2])
        

    feat_matrix["mass_center_x"] = mass_center_x
    feat_matrix["mass_center_y"] = mass_center_y
    feat_matrix["mass_center_z"] = mass_center_z
    feat_matrix["volume"] = volume
    feat_matrix["min_distance"] = min_distance
    feat_matrix["mean_distance"] = mean_distance
    feat_matrix["max_distance"] = max_distance
    feat_matrix["variance_x"] = variance_x
    feat_matrix["variance_y"] = variance_y
    feat_matrix["variance_z"] = variance_z
    
    return feat_matrix
 
def merge_df(feat_atom_mat, feat_mat):
    
    """The function performs a merge on two pandas tables
    
    Requirements:
        pandas
    
    Args:
        feat_atom_mat (pandas dataframe): First feature matrix
        feat_mat (pandas dataframe): Second feature matrix
        
    Remark:
        The ids have to match entry wise
    
    Returns:
        pandas dataframe full: Merged feature matrix
        
    """
    
    feat_atom_mat = feat_atom_mat.fillna(0)
    full = pd.concat([feat_mat, feat_atom_mat], axis=1, join_axes=[feat_mat.index])
    full = full.drop("id", axis = 1)
    
    return full
    
if __name__ == "__main__":
    
    # load and prepare data
    print("Load data")
    train = pd.read_csv("../input/train.txt")
    test = pd.read_csv("../input/test.txt")

    train_id = train["id"]
    test_id = test["id"]
    label = train[["formation_energy_ev_natom", "bandgap_energy_ev"]]
    train = train.drop(["id", "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)
    test = test.drop("id", axis = 1)
    
    # one hot encode of spacegroup
    print("One hot encode of 'Spacegroup' column")
    train = one_hot(train)
    test = one_hot(test)
    
    n_train = len(train)
    n_test = len(test)
    
    # load xyz-files
    print("Load geomatry (.xyz) files")
    train_atoms = pd.DataFrame(columns=list('ABCDE'))
    train_lattices = pd.DataFrame(columns=list('ABCE'))
    for k in range(n_train):

        idx = train.id.values[k]
        fn = "../input/train/{}/geometry.xyz".format(idx)
        train_xyz, train_lat = get_xyz_data(fn, k+1)
        train_atoms = train_atoms.append(train_xyz)
        train_lattices = train_lattices.append(train_lat)
    
    test_atoms = pd.DataFrame(columns=list('ABCDE'))
    test_lattices = pd.DataFrame(columns=list('ABCE'))
    for k in range(n_test):

        idx = test.id.values[k]
        fn = "../input/test/{}/geometry.xyz".format(idx)
        test_xyz, test_lat = get_xyz_data(fn, k+1)
        test_atoms = test_atoms.append(test_xyz)
        test_lattices = test_lattices.append(test_lat)
    
    # feature engineering with ase library
    print("Perform feature extraction on .xyz files")
    feat_matrix_train = feature_extraction(train_atoms, train_lattices, n_train)
    feat_matrix_test = feature_extraction(test_atoms, test_lattices, n_test)
    
    # merge data
    print("Merge data")
    train_full = merge_df(feat_matrix_train, train)
    test_full = merge_df(feat_matrix_test, test)
    
    # train - val split
    print("Train and validation split")
    X_train, X_val, y_train, y_val = train_test_split(train_full, label, test_size=0.2, random_state=RANDOM_STATE)
    
    # define models
    alg_1_a = XGBRegressor(learning_rate = 0.1, n_estimators=250, max_depth=4,
                           min_child_weight=1, gamma=0, subsample=0.857, colsample_bytree=0.8,
                           objective= 'reg:logistic', nthread=4, scale_pos_weight=1, seed=RANDOM_STATE)
    alg_2_a = XGBRegressor(learning_rate = 0.10, n_estimators=250, max_depth=4,
                            min_child_weight=1, gamma=0.08101, subsample=0.8,
                            colsample_bytree=0.8, objective= 'reg:linear',
                            nthread=4, scale_pos_weight=1, seed=RANDOM_STATE)
                           
    alg_1_b = GradientBoostingRegressor(max_depth=4, random_state=RANDOM_STATE, learning_rate=0.10, max_leaf_nodes=26)
    alg_2_b = GradientBoostingRegressor(max_depth=4, random_state=RANDOM_STATE, max_leaf_nodes=12)
    
    param_1_c = {'max_depth': 5, 'eta': 0.007, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse'}
    param_2_c = {'max_depth': 3, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse'}
    
    # train models and evaluate error
    print("Train XGBRegressor Model for formation_energy_ev_natom")
    alg_1_a.fit(X_train, y_train["formation_energy_ev_natom"])
    pred_val_1_a = alg_1_a.predict(X_val)
    pred_test_1_a = alg_1_a.predict(test_full)
    pred_train_1_a = alg_1_a.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_a, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_a, label["formation_energy_ev_natom"])))
    
    print("Train GradientBoostingRegressor Model for formation_energy_ev_natom")
    alg_1_b.fit(X_train, y_train["formation_energy_ev_natom"])
    pred_val_1_b = alg_1_b.predict(X_val)
    pred_test_1_b = alg_1_b.predict(test_full)
    pred_train_1_b = alg_1_b.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_b, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_b, label["formation_energy_ev_natom"])))
    
    print("Train xgboost Model for formation_energy_ev_natom")
    alg_1_c = xgb.train(param_1_c, xgb.DMatrix(X_train, label=y_train["formation_energy_ev_natom"]), num_boost_round = 2000)
    pred_val_1_c = alg_1_c.predict(X_val)
    pred_test_1_c = alg_1_c.predict(test_full)
    pred_train_1_c = alg_1_c.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_c, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_c, label["formation_energy_ev_natom"])))

    print("Train XGBRegressor Model for bandgap_energy_ev")
    alg_2_a.fit(X_train, y_train["bandgap_energy_ev"])
    pred_val_2_a = alg_2_a.predict(X_val)
    pred_test_2_a = alg_2_a.predict(test_full)
    pred_train_2_a = alg_2_a.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_2_b, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_2_b, label["bandgap_energy_ev"])))
    
    print("Train GradientBoostingRegressor Model for bandgap_energy_ev")
    alg_2_b.fit(X_train, y_train["bandgap_energy_ev"])
    pred_val_2_b = alg_2_b.predict(X_val)
    pred_test_2_b = alg_2_b.predict(test_full)
    pred_train_2_b = alg_2_b.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_2_b, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_2_b, label["bandgap_energy_ev"])))
    
    print("Train xgboost Model for bandgap_energy_ev")
    alg_2_c = xgb.train(param_2_c, xgb.DMatrix(X_train, label=y_train["bandgap_energy_ev"]), num_boost_round = 2000)    
    pred_val_2_c = alg_2_c.predict(X_val)
    pred_test_2_c = alg_2_c.predict(test_full)
    pred_train_2_c = alg_2_c.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_2_c, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_2_c, label["bandgap_energy_ev"])))
    
    # ensemble
    res_1 = [1./3., 1./3., 1./3.]
    res_2 = [1./3., 1./3., 1./3.]
    pred_test_1 = res_1[0]*pred_test_1_a + res_1[1]*pred_test_1_b + res_1[2]*pred_test_1_c
    pred_val_1 = res_1[0]*pred_val_1_a + res_1[1]*pred_val_1_b + res_1[2]*pred_val_1_c
    pred_train_1 = res_1[0]*pred_train_1_a + res_1[1]*pred_train_1_b + res_1[2]*pred_train_1_c
    val_error_1 = rmsle(pred_val_1, y_val["formation_energy_ev_natom"])
    train_error_1 = rmsle(pred_train_1, label["formation_energy_ev_natom"])
    print("RMSLE for validation data on formation_energy_ev_natom after ensemble: " + str(val_error_1))
    print("RMSLE for complete train data on formation_energy_ev_natom after ensemble: " + str(train_error_1))
    
    pred_test_2 = res_2.x[0]*pred_test_2_a + res_2.x[1]*pred_test_2_b + res_2.x[2]*pred_test_2_c
    pred_val_2 = res_2.x[0]*pred_val_2_a + res_2.x[1]*pred_val_2_b + res_2.x[2]*pred_val_2_c
    pred_train_2 = res_2[0]*pred_train_2_a + res_2[1]*pred_train_2_b + res_2[2]*pred_train_2_c
    val_error_2 = rmsle(pred_val_2, y_val["bandgap_energy_ev"])
    train_error_2 = rmsle(pred_train_2, label["bandgap_energy_ev"])
    print("RMSLE for validation data on bandgap_energy_ev after ensemble: " + str(val_error_2))
    print("RMSLE for complete train data on bandgap_energy_ev after ensemble: " + str(train_error_2))
    
    print("Total validation error (formation_energy_ev_natom and bandgap_energy_ev): " + str((val_error_1 + val_error_2)/2.0))
    print("Total train error: (formation_energy_ev_natom and bandgap_energy_ev): " + str((train_error_1 + train_error_2)/2.0))
    
    # save submission
    print("Save submission")
    save_submission(pred_test_1, pred_test_2, test_id)
    
    print("Finished")