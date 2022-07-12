import random
import numpy as np
import pandas as pd
from math import exp
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

def compute_beta(row):
    g = 1.0
    k=5.0
    f=1.0
    cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
    return 1.0 / (g + exp((cnt - k) / f))
        

def calculate_average(sub1, sub2, variable):
    r_k=0.01
    lambda_val = None
    s = pd.DataFrame(data = {
                             variable: sub1.groupby(variable, as_index = False).count()[variable],                              
                             'sumy': sub1.groupby(variable, as_index = False).sum()['y'],
                             'avgY': sub1.groupby(variable, as_index = False).mean()['y'],
                             'cnt': sub1.groupby(variable, as_index = False).count()['y']
                             })

    tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable) 
    del tmp['index']                       
    tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
    tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0



    if lambda_val is not None:
        tmp['beta'] = lambda_val
    else:
        tmp['beta'] = tmp.apply(compute_beta, axis = 1)

    tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],
                               axis = 1)

    tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
    tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
    tmp['random'] = np.random.uniform(size = len(tmp))
    tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] *(1 + (row['random'] - 0.5) * r_k),
                               axis = 1)

    return tmp['adj_avg'].ravel()
    

def categorical_average(variable, y, pred_0, feature_name):
    #cv for training set 
    k_fold = StratifiedKFold(5, random_state=myseed)
    X_train[feature_name] = -999 
    for (train_index, cv_index) in k_fold.split(np.zeros(len(X_train)),
                                                X_train['interest_level'].ravel()):
        sub = pd.DataFrame(data = {variable: X_train[variable],
                                   'y': X_train[y],
                                   'pred_0': X_train[pred_0]})
            
        sub1 = sub.iloc[train_index]        
        sub2 = sub.iloc[cv_index]
        
        X_train.loc[cv_index, feature_name] = calculate_average(sub1, sub2, variable)
    
    #for test set
    sub1 = pd.DataFrame(data = {variable: X_train[variable],
                                'y': X_train[y],
                                'pred_0': X_train[pred_0]})
    sub2 = pd.DataFrame(data = {variable: X_test[variable],
                                'y': X_test[y],
                                'pred_0': X_test[pred_0]})
    X_test.loc[:, feature_name] = calculate_average(sub1, sub2, variable)                               


def transform_data(X):
    #add features    
    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([ pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(), X1.reset_index()], axis = 1)
    del X['index']
    
    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed'] = X['price'] / X['bedrooms']    
    X['price_per_bath'] = X['price'] / X['bathrooms']
    X['price_per_room'] = X['price'] / (X['bathrooms'] + X['bedrooms'] )
    
    X['low'] = 0
    X.loc[X['interest_level'] == 0, 'low'] = 1
    X['medium'] = 0
    X.loc[X['interest_level'] == 1, 'medium'] = 1
    X['high'] = 0
    X.loc[X['interest_level'] == 2, 'high'] = 1
    
    X['display_address'] = X['display_address'].apply(lambda x: x.lower().strip())
    X['street_address'] = X['street_address'].apply(lambda x: x.lower().strip())
    
    X['pred0_low'] = low_count * 1.0 / train_size
    X['pred0_medium'] = medium_count * 1.0 / train_size
    X['pred0_high'] = high_count * 1.0 / train_size
    
    X.loc[X['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()), 
          'manager_id'] = "-1"
    X.loc[X['building_id'].isin(buildings_with_one_lot['building_id'].ravel()), 
          'building_id'] = "-1"
    X.loc[X['display_address'].isin(addresses_with_one_lot['display_address'].ravel()), 
          'display_address'] = "-1"
          
    return X


def normalize_high_cardinality_data():
    high_cardinality = ["building_id", "manager_id"]
    for c in high_cardinality:
        categorical_average(c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(c, "high", "pred0_high", c + "_mean_high")

        
def transform_categorical_data():
    categorical = ['building_id', 'manager_id', 
                   'display_address', 'street_address']
                   
    for f in categorical:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[f]) + list(X_test[f])) 
        X_train[f] = encoder.transform(X_train[f].ravel())
        X_test[f] = encoder.transform(X_test[f].ravel())
                  

def remove_columns(X):
    columns = ["photos", "pred0_high", "pred0_low", "pred0_medium",
               "description", "low", "medium", "high",
               "interest_level", "created"]
    for c in columns:
        del X[c]

        
def find_objects_with_only_one_record(feature_name):
    temp = pd.concat([X_train[feature_name].reset_index(), 
                      X_test[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]


def Outputs(p):
    return (1./(1.+np.exp(-p)))


def GPNotLo1(data):
    p = (-0.822105 +
        0.100000*np.tanh(((data["furnished"] + ((data["bedrooms"] + (data["no_fee"] + (((data["manager_id_mean_medium"] + np.tanh((data["reduced_fee"] + (data["manager_id_mean_high"] + data["building_id_mean_high"])))) - data["price"]) * 2.0))) * 2.0)) * 2.0)) +
        0.100000*np.tanh((((((data["building_id_mean_medium"] + (np.tanh((((data["building_id_mean_high"] * 2.0) * 2.0) * 2.0)) + (data["manager_id_mean_high"] + data["manager_id_mean_medium"]))) * 2.0) + data["manager_id_mean_medium"]) + (data["reduced_fee"] + data["no_fee"])) * 2.0)) +
        0.100000*np.tanh((data["furnished"] + (((((data["building_id_mean_medium"] + ((data["manager_id_mean_medium"] * 2.0) + (data["bedrooms"] - data["price"]))) - (0.538462 + data["price"])) + data["building_id_mean_high"]) * 2.0) * 2.0))) +
        0.100000*np.tanh((((((data["manager_id_mean_medium"] - (data["price"] - ((data["manager_id_mean_high"] + (data["no_fee"] - (data["price_per_room"] + (data["price"] + ((data["price_per_room"] > data["building_id"]).astype(float))))))/2.0))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((6.0) * (((((((data["building_id_mean_medium"] + ((data["building_id_mean_medium"] / 2.0) * np.tanh(data["manager_id_mean_medium"])))/2.0) + np.tanh((data["manager_id_mean_medium"] + data["actual_apt"]))) * 2.0) - data["price"]) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["laundry_in_unit"] + ((((data["building_id_mean_medium"] + data["laundry_in_building"]) - (data["price"] * 2.0)) + (data["manager_id_mean_high"] + (data["bedrooms"] - (((data["price"] > data["bedrooms"]).astype(float)) - data["manager_id_mean_medium"])))) * 2.0))) +
        0.100000*np.tanh(((data["no_fee"] + ((data["private_outdoor_space"] + (((data["reduced_fee"] + data["manager_id_mean_high"]) - data["price"]) * 2.0)) + (((data["manager_id_mean_medium"] + data["furnished"])/2.0) + (data["bedrooms"] + data["building_id_mean_medium"])))) * 2.0)) +
        0.100000*np.tanh(((((data["reduced_fee"] - (data["price"] - ((data["manager_id_mean_medium"] + (data["manager_id_mean_high"] + ((data["building_id_mean_high"] + (data["bedrooms"] + np.tanh(data["building_id_mean_medium"]))) + -1.0)))/2.0))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["hardwood_floors"] / 2.0) + (data["manager_id_mean_high"] + (data["furnished"] - ((((data["price"] + ((data["price_per_room"] > data["no_fee"]).astype(float)))/2.0) * 2.0) * 2.0)))) + (data["manager_id_mean_medium"] + data["building_id_mean_high"])) * 2.0)) +
        0.100000*np.tanh((((data["hardwood_floors"] + ((data["light"] + ((((data["manager_id_mean_high"] + ((data["manager_id_mean_medium"] + ((data["building_id_mean_medium"] > data["live"]).astype(float))) * 2.0)) * 2.0) * 2.0) + data["building_id_mean_high"])) * 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["manager_id_mean_medium"] + (((data["manager_id_mean_medium"] > data["manager_id_mean_high"]).astype(float)) + (data["furnished"] + ((data["building_id_mean_medium"] + ((data["manager_id_mean_high"] - ((data["elevator"] > (-(data["price_per_bed"]))).astype(float))) * 2.0)) * 2.0))))) +
        0.100000*np.tanh((((((data["laundry_in_unit"] - ((data["price"] - data["reduced_fee"]) - np.tanh(np.tanh(data["building_id_mean_high"])))) * 2.0) + (data["manager_id_mean_high"] - ((-(data["no_fee"])) - data["laundry_in_building"]))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["common_outdoor_space"] + (data["bedrooms"] + (data["building_id_mean_medium"] - (((data["manager_id_mean_high"] > data["num_photos"]).astype(float)) - data["manager_id_mean_high"])))) + ((-((data["price"] - (data["hardwood_floors"] / 2.0)))) * 2.0))) +
        0.100000*np.tanh(((data["manager_id_mean_medium"] + (np.tanh((((data["price_per_bath"] + (np.tanh(((data["building_id_mean_medium"] + (((data["building_id_mean_high"] + data["furnished"]) * 2.0) * 2.0)) * 2.0)) * 2.0)) * 2.0) * 2.0)) * 2.0)) * 2.0)) +
        0.100000*np.tanh((((data["manager_id_mean_medium"] + (((((-((data["price"] - data["bedrooms"]))) * 2.0) - ((data["laundry_in_unit"] < (data["bedrooms"] / 2.0)).astype(float))) + (data["laundry_in_building"] / 2.0)) * 2.0)) - data["simplex"]) * 2.0)) +
        0.100000*np.tanh(((data["furnished"] - (-((data["no_fee"] - (-((data["manager_id_mean_high"] - ((data["price"] + ((data["reduced_fee"] < (data["price"] + ((data["post_war"] < data["longitude"]).astype(float)))).astype(float))) * 2.0)))))))) * 2.0)) +
        0.100000*np.tanh(((-((((data["price"] - data["bedrooms"]) * 2.0) * 2.0))) - (data["lowrise"] - ((((data["private_outdoor_space"] + data["manager_id_mean_medium"]) + data["price_per_room"]) + (-(data["listing_id"]))) + data["no_fee"])))) +
        0.100000*np.tanh((data["manager_id_mean_high"] + ((data["laundry_in_building"] - (data["price"] / 2.0)) + ((data["reduced_fee"] - (data["price"] + (((-(data["site_laundry"])) > ((data["furnished"] + data["price"]) / 2.0)).astype(float)))) * 2.0)))) +
        0.100000*np.tanh(((((data["no_fee"] + (((data["building_id_mean_medium"] + data["manager_id_mean_high"]) + (((-((0.708861 - ((data["children"] > (data["longitude"] * 2.0)).astype(float))))) * 2.0) * 2.0)) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((((data["num_photos"] * (-(data["num_photos"]))) + np.tanh((2.875000 - (-((((-((data["doorman"] * data["building_id_mean_medium"]))) * 2.0) * 2.0)))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["price_per_bed"] + (1.545450 + (data["num_photos"] - ((((((data["longitude"] > (-((((data["pet_friendly"] < data["price_per_bed"]).astype(float)) * 2.0)))).astype(float)) * 2.0) * 2.0) * 2.0) * 2.0)))) * 2.0) * 2.0)) +
        0.100000*np.tanh((((-(((-(data["bedrooms"])) + (data["price"] - (data["building_id"] / 2.0))))) + data["laundry_in_building"]) - (((data["display_address"] + data["price"]) - data["furnished"]) - data["private_outdoor_space"]))) +
        0.100000*np.tanh((((((np.tanh(np.tanh(data["price"])) + ((data["pool"] > ((data["publicoutdoor"] * data["price"]) * 2.0)).astype(float))) - data["price"]) - (data["num_photos"] * (data["num_photos"] / 2.0))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((((-((((data["longitude"] > ((-((data["common_outdoor_space"] * data["full_service_garage"]))) * 2.0)).astype(float)) * 2.0))) + data["common_outdoor_space"]) - data["simplex"]) - data["price"]) + data["num_photos"]) - data["fitness_center"])) +
        0.100000*np.tanh(((((data["no_fee"] + ((data["price_per_bath"] + (((data["no_fee"] - data["price_per_room"]) + ((((data["reduced_fee"] > (data["price_per_room"] * 2.0)).astype(float)) * 2.0) + data["manager_id_mean_medium"])) * 2.0))/2.0)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((((data["renovated"] - data["price"]) - ((data["video_intercom"] < (data["longitude"] * (2.875000 - (data["price_per_room"] * data["price_per_room"])))).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["courtyard"] > (data["valet"] + data["latitude"])).astype(float)) + (((data["num_photos"] - ((data["view"] > data["latitude"]).astype(float))) - ((data["view"] > data["latitude"]).astype(float))) - ((data["view"] > data["latitude"]).astype(float))))) +
        0.100000*np.tanh((((((data["view"] < data["latitude"]).astype(float)) * 2.0) - ((np.tanh(((data["longitude"] > (-(np.tanh((data["latitude"] - (data["residents_garden"] / 2.0)))))).astype(float))) * 2.0) * 2.0)) - data["price"])) +
        0.100000*np.tanh((data["bedrooms"] + (data["dogs_allowed"] + (data["no_fee"] - ((((data["simplex"] * 2.0) * 2.0) + data["price_per_bed"]) - (data["num_description_words"] - ((data["price"] * 2.0) - data["bedrooms"]))))))) +
        0.100000*np.tanh((data["furnished"] + ((data["bathrooms"] + (data["building_id"] * (-(data["building_id_mean_medium"])))) - (((data["no_pets"] > data["latitude"]).astype(float)) - ((data["street_address"] * (-(data["building_id_mean_medium"]))) - data["price"]))))) +
        0.100000*np.tanh(((data["building_id"] + data["common_outdoor_space"]) + ((data["exclusive"] + (data["reduced_fee"] + (data["renovated"] + data["swimming_pool"]))) - np.tanh(((((data["in_superintendent"] + data["street_address"]) * 2.0) * 2.0) * 2.0))))) +
        0.100000*np.tanh((((data["doorman"] * ((data["microwave"] < (data["longitude"] * 2.0)).astype(float))) - (data["simplex"] - (data["hardwood_floors"] * (((data["hardwood_floors"] * ((data["high_ceilings"] > data["hardwood_floors"]).astype(float))) > data["manager_id_mean_medium"]).astype(float))))) * 2.0)) +
        0.100000*np.tanh((data["furnished"] + ((data["private_outdoor_space"] / 2.0) - (data["building_id_mean_medium"] - (data["bedrooms"] - (((data["price"] + (data["building_id_mean_medium"] * ((data["building_id"] - data["war"]) - data["simplex"])))/2.0) * 2.0)))))) +
        0.100000*np.tanh((((((data["short_term_allowed"] + ((((data["level"] > (data["manager_id_mean_medium"] - ((data["exclusive"] > data["building_id_mean_high"]).astype(float)))).astype(float)) + (data["building_id_mean_medium"] * ((data["manager_id_mean_high"] > data["manager_id_mean_medium"]).astype(float))))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["manager_id_mean_medium"] + (((((data["site_garage"] < ((data["dryer"] - data["unit_washer"]) - data["latitude"])).astype(float)) - ((((data["latitude"] < (data["dryer"] - data["full_service_garage"])).astype(float)) * 2.0) * 2.0)) * 2.0) * 2.0))/2.0)) +
        0.100000*np.tanh((data["price_per_bath"] - (-(((data["furnished"] + (-(((((data["price"] + ((data["longitude"] + (data["longitude"] + ((data["longitude"] + 1.0)/2.0))) * 2.0)) * 2.0) * 2.0) * 2.0))))/2.0))))) +
        0.099990*np.tanh(((((-((((data["dishwasher"] < data["hardwood_floors"]).astype(float)) - (data["num_photos"] + ((data["listing_id"] < (1.0 + ((data["laundry_in_building"] < (data["manager_id"] / 2.0)).astype(float)))).astype(float)))))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["laundry_in_building"] * (data["dogs_allowed"] + np.tanh(((-(data["bedrooms"])) * 2.0)))) + (((-(((data["wheelchair_ramp"] > ((data["childrens_playroom"] + (data["basement_storage"] - data["longitude"]))/2.0)).astype(float)))) * 2.0) * 2.0))) +
        0.096200*np.tanh((((((-((data["no_fee"] * ((data["num_description_words"] - (-(((data["building_id_mean_medium"] * (data["prewar"] * data["building_id_mean_medium"])) * 2.0)))) + (data["cats_allowed"] / 2.0))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["wheelchair_access"] + (((data["actual_apt"] + (data["manager_id_mean_high"] / 2.0))/2.0) + ((data["exclusive"] + ((((data["manager_id_mean_medium"] + data["parking_space"])/2.0) > np.tanh(data["manager_id_mean_high"])).astype(float))) * 2.0))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((-((((data["high_speed_internet"] + ((data["simplex"] > np.tanh(((data["exclusive"] * data["building_id_mean_medium"]) - (data["building_id_mean_medium"] + data["manager_id_mean_medium"])))).astype(float))) * 2.0) * 2.0))) + (data["building_id_mean_medium"] + data["manager_id_mean_medium"]))) +
        0.100000*np.tanh(((data["laundry_in_unit"] - (1.47503650188446045)) - (data["building_id_mean_medium"] - (((data["bedrooms"] - (data["price"] - ((((data["cats_allowed"] + data["bathrooms"])/2.0) > (data["bedrooms"] - data["bathrooms"])).astype(float)))) * 2.0) * 2.0)))) +
        0.100000*np.tanh((((((data["new_construction"] > (-(((data["longitude"] > (data["shares_ok"] / 2.0)).astype(float))))).astype(float)) < ((data["shares_ok"] < data["bedrooms"]).astype(float))).astype(float)) - ((data["longitude"] > (data["decorative_fireplace"] * ((data["bathrooms"] > data["private_outdoor_space"]).astype(float)))).astype(float)))) +
        0.100000*np.tanh(((-(((data["live"] * 2.0) * 2.0))) - ((data["price"] / 2.0) - (-(((data["longitude"] < (data["highrise"] + (((data["common_roof_deck"] * (data["price"] / 2.0)) < data["longitude"]).astype(float)))).astype(float))))))) +
        0.100000*np.tanh((-((((((0.538462 - data["bedrooms"]) + data["num_photos"])/2.0) + (data["price"] - ((data["price"] > data["manager_id_mean_medium"]).astype(float)))) - ((data["parking_space"] - ((data["num_photos"] < -1.0).astype(float))) * 2.0))))) +
        0.095330*np.tanh(np.tanh(((14.57628250122070312) * ((14.57628250122070312) * ((14.57628250122070312) * (data["24"] - (data["manager_id_mean_medium"] * (-(((data["furnished"] - (-(data["laundry_room"]))) + (-(data["manager_id_mean_high"])))))))))))) +
        0.092520*np.tanh((((((data["longitude"] < (-(((data["prewar"] - data["building_id_mean_medium"]) * (data["longitude"] - (((data["manager_id_mean_medium"] / 2.0) / 2.0) / 2.0)))))).astype(float)) + (data["fitness_center"] / 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh(((-(((data["fitness_center"] * (np.tanh((-(data["pre"]))) + data["manager_id_mean_high"])) + ((data["exclusive"] > (data["pre"] * ((data["listing_id"] + ((data["simplex"] + data["exclusive"])/2.0))/2.0))).astype(float))))) * 2.0)) +
        0.100000*np.tanh((data["price"] - (((((data["duplex"] > data["latitude"]).astype(float)) + (data["latitude"] + (-((((data["green_building"] > ((data["price"] + ((data["eat_in_kitchen"] < data["building_id_mean_high"]).astype(float)))/2.0)).astype(float)) * 2.0))))) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["_photos"] - (data["green_building"] + (3.666670 * (data["prewar"] + ((data["longitude"] > (data["actual_apt"] * (data["patio"] + ((data["price"] + (data["listing_id"] * data["listing_id"]))/2.0)))).astype(float))))))) +
        0.100000*np.tanh((data["bedrooms"] * (data["site_garage"] - ((-((data["terrace"] + (data["concierge"] + ((data["building_id_mean_high"] > (((data["fitness_center"] + data["prewar"])/2.0) + ((data["building_id_mean_high"] + data["no_fee"])/2.0))).astype(float)))))) * 2.0)))) +
        0.100000*np.tanh((data["building_id_mean_high"] + (data["building_id_mean_high"] + (((-(((data["latitude"] < data["walk"]).astype(float)))) * 2.0) + ((((data["building_id_mean_high"] < ((-1.0 - data["garage"]) - data["exclusive"])).astype(float)) * 2.0) * 2.0))))) +
        0.100000*np.tanh((data["site_laundry"] - (-((((data["price_per_bed"] < data["green_building"]).astype(float)) - (-((data["outdoor_space"] - (-((data["_photos"] - (data["price_per_bed"] * (-(((data["price_per_bed"] < data["no_fee"]).astype(float)))))))))))))))) +
        0.100000*np.tanh((data["reduced_fee"] + ((data["storage"] - ((data["listing_id"] > 1.543480).astype(float))) + (-(((data["price"] - (data["renovated"] - data["subway"])) + ((data["price"] > data["listing_id"]).astype(float)))))))) +
        0.100000*np.tanh((-1.0 - ((data["bathrooms"] - ((np.tanh(data["stainless_steel_appliances"]) - ((data["bathrooms"] - ((((data["prewar"] < ((-1.0 + data["bathrooms"])/2.0)).astype(float)) * 2.0) * 2.0)) / 2.0)) * 2.0)) * 2.0))) +
        0.100000*np.tanh((data["private_outdoor_space"] + ((-(((data["latitude"] > (data["fitness_center"] * (data["longitude"] + (data["longitude"] - (data["post"] * 2.0))))).astype(float)))) + (data["fitness_center"] * (data["fitness_center"] - data["num_description_words"]))))) +
        0.096250*np.tanh(((((np.tanh(np.tanh((((data["furnished"] * ((data["wheelchair_access"] < (data["building_id_mean_medium"] + (data["fitness_center"] * 2.0))).astype(float))) * 2.0) * 2.0))) * 2.0) * 2.0) * 2.0) + (data["building_id_mean_medium"] * data["dogs_allowed"]))) +
        0.100000*np.tanh((((data["longitude"] < (data["decorative_fireplace"] - ((data["latitude"] + (((data["longitude"] / 2.0) > data["latitude"]).astype(float)))/2.0))).astype(float)) - ((data["price"] + (data["num_photos"] * (data["num_photos"] * (-(data["laundry_in_building"])))))/2.0))) +
        0.100000*np.tanh((data["bathrooms"] + (data["bedrooms"] - ((data["washer_in_unit"] + (data["concierge"] + (data["outdoor_areas"] + ((-((data["hardwood_floors"] * data["hardwood"]))) + (data["price"] / 2.0))))) * 2.0)))) +
        0.100000*np.tanh(((data["doorman"] * ((data["loft"] > (data["price"] + data["no_fee"])).astype(float))) - (((data["no_fee"] - data["price"]) * data["manager_id_mean_medium"]) * data["price"]))) +
        0.100000*np.tanh((((((-(data["playroom"])) > (data["latitude"] * data["hardwood_floors"])).astype(float)) + (-1.0 / 2.0)) - ((data["longitude"] > ((data["latitude"] + ((data["roof"] * data["hardwood_floors"]) / 2.0))/2.0)).astype(float)))) +
        0.100000*np.tanh((((0.439024 < ((-((data["manager_id_mean_high"] * 2.0))) / 2.0)).astype(float)) - ((((data["bathrooms"] / 2.0) > ((0.439024 < (data["bathrooms"] / 2.0)).astype(float))).astype(float)) - data["_photos"]))) +
        0.100000*np.tanh((data["common_roof_deck"] + (data["actual_apt"] + ((0.40919432044029236) - ((((-(data["latitude"])) * 2.0) < ((data["outdoor"] + (data["virtual_doorman"] + (data["bathrooms"] * ((data["longitude"] > data["site_parking_lot"]).astype(float)))))/2.0)).astype(float)))))) +
        0.100000*np.tanh((-((((data["building_id_mean_high"] > ((data["building_id_mean_medium"] + (((data["roof_deck"] * (data["deck"] + data["building_id_mean_high"])) < (data["deck"] + data["building_id_mean_medium"])).astype(float)))/2.0)).astype(float)) + (data["building_id_mean_high"] * (data["roof_deck"] / 2.0)))))) +
        0.099990*np.tanh(((-((data["display_address"] * ((data["exclusive"] + (data["roof_deck"] + (data["num_photos"] + (data["_pets_ok_"] + (data["exclusive"] + ((data["in_superintendent"] - data["laundry_in_building"]) - data["concierge"]))))))/2.0)))) * 2.0)) +
        0.100000*np.tanh((((((data["prewar"] * data["manager_id_mean_medium"]) + (data["building_id_mean_medium"] * (data["manager_id_mean_high"] * data["manager_id_mean_medium"]))) * 2.0) * 2.0) + (np.tanh((data["building_id_mean_medium"] * (data["listing_id"] - data["high_speed_internet"]))) * 2.0))) +
        0.100000*np.tanh(((((data["building_id_mean_medium"] * data["prewar"]) - ((data["washer_in_unit"] > (data["listing_id"] * data["outdoor_areas"])).astype(float))) - ((data["pool"] * 2.0) * 2.0)) - ((data["washer_in_unit"] > (data["latitude"] - data["highrise"])).astype(float)))) +
        0.100000*np.tanh((data["no_fee"] * (data["common_outdoor_space"] + ((data["building_id"] + ((((((data["latitude"] > data["luxury_building"]).astype(float)) + (-((((data["latitude"] + data["_photos"]) < data["_photos"]).astype(float)))))/2.0) * 2.0) * 2.0))/2.0)))) +
        0.100000*np.tanh(((data["high_ceiling"] + (((data["furnished"] * 2.0) * 2.0) + (data["laundry"] + ((data["latitude"] > (data["bedrooms"] * data["laundry"])).astype(float))))) - ((data["childrens_playroom"] > ((-(data["latitude"])) / 2.0)).astype(float)))) +
        0.100000*np.tanh((-((((data["high_speed_internet"] * 2.0) + (((-(((data["price_per_bath"] < ((data["display_address"] < data["price_per_room"]).astype(float))).astype(float)))) < ((data["bedrooms"] + data["price"])/2.0)).astype(float))) * 2.0)))) +
        0.100000*np.tanh(((data["street_address"] * (data["central_ac"] - ((data["patio"] - data["price_per_bath"]) + (data["building_id_mean_medium"] - data["level"])))) - ((data["listing_id"] * 2.0) * ((3.571430 < (data["listing_id"] * 2.0)).astype(float))))) +
        0.100000*np.tanh(((((data["wheelchair_access"] + (((data["wheelchair_access"] + ((-(data["bathrooms"])) + (((data["bathrooms"] > ((data["washer_in_unit"] < data["bathrooms"]).astype(float))).astype(float)) * 2.0)))/2.0) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["common_roof_deck"] + ((((np.tanh((data["price_per_bath"] * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) * ((data["balcony"] > (data["price_per_bath"] * (data["num_description_words"] * ((data["price_per_bath"] + data["laundry_in_building"])/2.0)))).astype(float)))) +
        0.100000*np.tanh((data["roof_deck"] * (data["loft"] - (-((data["lowrise"] - (-((((data["dishwasher"] < (data["building_id_mean_high"] + (((0.538462 - 0.538462) + data["cats_allowed"])/2.0))).astype(float)) * 2.0))))))))) +
        0.100000*np.tanh((((data["latitude"] < (-((data["green_building"] * (np.tanh(np.tanh(((data["lounge_room"] > data["longitude"]).astype(float)))) / 2.0))))).astype(float)) - ((data["simplex"] + data["simplex"]) * (-(data["building_id_mean_medium"]))))) +
        0.100000*np.tanh(((data["hardwood_floors"] - (-(data["building_id_mean_medium"]))) * (((((-(data["live"])) > (((data["publicoutdoor"] + ((data["terrace"] + (-(data["building_id_mean_medium"])))/2.0))/2.0) / 2.0)).astype(float)) < (data["manager_id_mean_high"] * data["building_id"])).astype(float)))) +
        0.100000*np.tanh(np.tanh(((((((-(((data["price"] + ((data["high_ceiling"] > (data["simplex"] + (data["latitude"] * 2.0))).astype(float))) * 2.0))) + ((data["no_pets"] < data["price"]).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh((((((((data["longitude"] < (-((((data["bathrooms"] < 0.538462).astype(float)) * 2.0)))).astype(float)) > (((1.545450 + (data["price"] - data["short_term_allowed"]))/2.0) / 2.0)).astype(float)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["exclusive"] * data["manager_id_mean_high"]) + (((data["manager_id_mean_high"] * 2.0) * 2.0) * ((0.144330 < (data["manager_id_mean_medium"] * data["fireplace"])).astype(float)))) + (data["price_per_bath"] + (data["manager_id_mean_high"] * data["building_id_mean_high"])))) +
        0.100000*np.tanh((data["bathrooms"] - ((data["bathrooms"] > (data["listing_id"] * (((np.tanh((((data["display_address"] - data["multi"]) - data["longitude"]) - data["listing_id"])) / 2.0) - data["multi"]) / 2.0))).astype(float)))) +
        0.100000*np.tanh(((((data["bike_room"] + (((data["price"] < ((-(3.571430)) / 2.0)).astype(float)) + (-((data["longitude"] * (data["private_outdoor_space"] - (-(3.571430)))))))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["building_id_mean_medium"] < (-(data["laundry_in_building"]))).astype(float)) + (-((((-((data["num_photos"] * data["swimming_pool"]))) < data["subway"]).astype(float))))) + (data["num_photos"] * data["dogs_allowed"]))) +
        0.100000*np.tanh(((-(((((data["latitude"] < data["highrise"]).astype(float)) + (data["listing_id"] * np.tanh(np.tanh((data["listing_id"] - (data["wheelchair_access"] + (data["display_address"] + ((-(data["manager_id"])) * 2.0))))))))/2.0))) * 2.0)) +
        0.100000*np.tanh(((data["bedrooms"] * ((data["parking_space"] < ((data["reduced_fee"] * np.tanh(data["display_address"])) * 2.0)).astype(float))) - (((((data["reduced_fee"] * data["parking_space"]) * 2.0) * 2.0) + data["price"]) + data["in_super"]))) +
        0.100000*np.tanh((data["short_term_allowed"] + (data["stainless_steel_appliances"] + (((data["fitness_center"] + (-(((data["common_outdoor_space"] - data["laundry_in_building"]) * data["doorman"]))))/2.0) + (data["fitness_center"] * (data["manager_id_mean_high"] - data["street_address"])))))) +
        0.100000*np.tanh(((((((((data["site_parking_lot"] + data["latitude"]) < np.tanh((data["site_parking"] - ((data["longitude"] + ((data["longitude"] > data["site_parking_lot"]).astype(float)))/2.0)))).astype(float)) > ((data["latitude"] < data["short_term_allowed"]).astype(float))).astype(float)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((-(((data["childrens_playroom"] > (-(((data["latitude"] / 2.0) + ((data["green_building"] > ((data["latitude"] + (data["publicoutdoor"] / 2.0)) / 2.0)).astype(float)))))).astype(float)))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((np.tanh((data["laundry_room"] * 2.0)) > (data["num_photos"] + 0.538462)).astype(float)) - (((((data["gym"] > (((data["num_photos"] + (0.538462 * 2.0))/2.0) / 2.0)).astype(float)) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["price_per_bed"] * (((data["num_description_words"] + data["cats_allowed"])/2.0) + ((((data["price_per_bed"] * (((data["new_construction"] + (data["newly_renovated"] + data["price_per_bath"])) + data["cats_allowed"])/2.0)) + data["private_outdoor_space"])/2.0) + data["patio"])))) +
        0.100000*np.tanh((((data["bathrooms"] + (-((data["manager_id_mean_high"] * (data["building_id"] + (data["display_address"] + data["building_id"]))))))/2.0) + (-((data["num_description_words"] * ((data["new_construction"] + data["stainless_steel_appliances"]) + data["furnished"])))))) +
        0.100000*np.tanh((data["price"] - (data["price"] + ((data["price_per_bath"] * data["bathrooms"]) - (-((((((data["_pets_ok_"] > (data["price"] + data["price_per_bath"])).astype(float)) > (data["price_per_bath"] - data["price"])).astype(float)) * 2.0))))))) +
        0.100000*np.tanh((-((((((data["price"] > data["price_per_bed"]).astype(float)) * data["building_id_mean_high"]) + ((data["price"] + data["concierge"])/2.0)) + ((((data["price"] > data["num_description_words"]).astype(float)) * data["building_id_mean_high"]) * data["building_id_mean_high"]))))) +
        0.100000*np.tanh(((((data["high_ceiling"] + data["bedrooms"])/2.0) - ((((((0.708861 < ((data["outdoor_areas"] + ((data["view"] + data["bedrooms"])/2.0))/2.0)).astype(float)) * 2.0) * 2.0) * 2.0) * 2.0)) + (data["hardwood"] * data["building_id"]))) +
        0.100000*np.tanh((data["midrise"] + (data["actual_apt"] + ((data["duplex"] + ((data["price_per_bath"] + ((((data["multi"] + ((data["multi"] + ((data["price_per_room"] > (-(data["dryer_in_building"]))).astype(float)))/2.0)) * 2.0) * 2.0) * 2.0))/2.0))/2.0)))) +
        0.099010*np.tanh(((data["price_per_room"] * (data["doorman"] * 2.0)) * (data["doorman"] - ((-(((data["price"] > (data["renovated"] + (data["virtual_doorman"] + (data["dishwasher"] * 2.0)))).astype(float)))) + data["dishwasher"])))) +
        0.095200*np.tanh((((data["bedrooms"] + (data["num_description_words"] * (data["no_fee"] - ((data["num_photos"] > (data["no_fee"] + data["no_fee"])).astype(float)))))/2.0) + (data["num_description_words"] * (data["building_id"] - (data["hardwood"] + data["outdoor_space"]))))) +
        0.100000*np.tanh(((-(((((data["longitude"] + data["site_garage"])/2.0) > (((data["site_garage"] + data["latitude"])/2.0) / 2.0)).astype(float)))) + (((data["building_id_mean_medium"] * (data["simplex"] + (-(data["common_outdoor_space"])))) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["price"] * ((-(((data["wheelchair_access"] < data["num_photos"]).astype(float)))) + (data["price_per_bed"] * (((-(((data["wheelchair_access"] < (data["num_photos"] / 2.0)).astype(float)))) * data["outdoor_space"]) * data["outdoor_space"]))))) +
        0.099990*np.tanh((data["_photos"] + (data["new_construction"] + (((data["price"] * ((((data["valet_parking"] > data["longitude"]).astype(float)) * 2.0) * 2.0)) + (data["price_per_bed"] - (data["price"] * (data["price"] * data["price"]))))/2.0)))) +
        0.099990*np.tanh(((((((data["dishwasher"] * (((data["video_intercom"] / 2.0) < data["longitude"]).astype(float))) * 2.0) * 2.0) * 2.0) * ((data["video_intercom"] < data["latitude"]).astype(float))) + ((np.tanh(data["video_intercom"]) < data["longitude"]).astype(float)))))
    return Outputs(p) 


def GPNotLo2(data):
    p = (-0.822105 +
        0.100000*np.tanh(((data["private_outdoor_space"] + (data["reduced_fee"] + (((data["no_fee"] + ((data["manager_id_mean_medium"] - (data["price"] - (data["building_id_mean_medium"] + data["manager_id_mean_high"]))) * 2.0)) * 2.0) + data["furnished"]))) - data["price_per_room"])) +
        0.100000*np.tanh((((((((np.tanh(np.tanh((data["manager_id_mean_medium"] + np.tanh((np.tanh(data["building_id_mean_high"]) + data["furnished"]))))) * 2.0) + data["manager_id_mean_high"]) - data["price"]) * 2.0) + data["building_id_mean_medium"]) * 2.0) * 2.0)) +
        0.100000*np.tanh((((((data["manager_id_mean_high"] + (data["manager_id_mean_high"] + (data["building_id_mean_medium"] + (data["no_fee"] - np.tanh(data["doorman"]))))) + ((np.tanh(data["building_id_mean_high"]) + data["manager_id_mean_medium"]) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["no_fee"] + data["building_id_mean_high"]) + data["building_id_mean_medium"]) + ((data["manager_id_mean_medium"] + (((data["manager_id_mean_high"] - data["price"]) + data["bedrooms"]) - (data["price"] + 0.621212))) * 2.0)) * 2.0)) +
        0.100000*np.tanh((((data["no_fee"] + ((data["bedrooms"] + (((np.tanh(np.tanh(data["building_id_mean_high"])) + data["manager_id_mean_medium"]) + (data["reduced_fee"] - data["price"])) * 2.0)) * 2.0)) + data["furnished"]) - data["lowrise"])) +
        0.100000*np.tanh((((((data["building_id_mean_high"] + ((data["renovated"] + (((data["post"] > data["price_per_bed"]).astype(float)) + ((data["reduced_fee"] * 2.0) * 2.0))) + (data["manager_id_mean_medium"] + data["manager_id_mean_high"]))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["manager_id_mean_medium"] - (data["price"] - ((((-(data["listing_id"])) + (data["manager_id_mean_high"] + -1.0))/2.0) + (data["building_id_mean_medium"] + (data["bedrooms"] - data["price"]))))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["no_fee"] + (((data["reduced_fee"] + (np.tanh(data["laundry_in_unit"]) + ((data["building_id_mean_medium"] + ((data["building_id_mean_high"] + data["laundry_in_building"])/2.0))/2.0))) + (data["manager_id_mean_high"] - data["price"])) * 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh((((((data["bedrooms"] - (data["price"] - ((data["laundry_in_building"] + (((data["laundry_in_unit"] + data["manager_id_mean_high"])/2.0) + np.tanh((data["building_id_mean_high"] * 2.0))))/2.0))) * 2.0) + data["manager_id_mean_medium"]) * 2.0) - data["bedrooms"])) +
        0.100000*np.tanh(((data["no_fee"] + (data["manager_id_mean_high"] + (data["furnished"] + (((data["building_id_mean_high"] + (data["manager_id_mean_medium"] - (data["price"] + ((0.621212 > data["bathrooms"]).astype(float))))) * 2.0) - data["price"])))) * 2.0)) +
        0.100000*np.tanh(((data["war"] + (((data["building_id_mean_medium"] + (((data["manager_id_mean_medium"] + np.tanh(((data["wifi_access"] < data["building_id_mean_medium"]).astype(float)))) * 2.0) * 2.0)) - ((data["num_photos"] < data["concierge"]).astype(float))) * 2.0)) + data["manager_id_mean_high"])) +
        0.100000*np.tanh((data["manager_id_mean_high"] + ((data["common_outdoor_space"] * 2.0) + ((data["hardwood_floors"] + ((data["building_id_mean_medium"] - data["price"]) + ((data["bedrooms"] - ((data["furnished"] < (1.66291987895965576)).astype(float))) - data["price"]))) * 2.0)))) +
        0.100000*np.tanh(((data["furnished"] + (data["laundry_in_unit"] + (data["reduced_fee"] + ((data["parking_space"] + ((data["manager_id_mean_high"] - ((data["view"] > (data["building_id_mean_high"] + data["reduced_fee"])).astype(float))) - data["price"])) * 2.0)))) * 2.0)) +
        0.100000*np.tanh((data["no_fee"] + (((data["num_photos"] + (data["manager_id_mean_medium"] + (data["no_fee"] - (((data["price"] - (np.tanh((data["bedrooms"] / 2.0)) * 2.0)) * 2.0) * 2.0)))) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["furnished"] + (((-(data["doorman"])) + (data["manager_id_mean_high"] + (((-(((((data["reduced_fee"] + data["building_id_mean_medium"]) * 2.0) < (-(data["building_id_mean_high"]))).astype(float)))) * 2.0) * 2.0))) * 2.0))) +
        0.100000*np.tanh((((data["laundry_in_building"] - (data["price"] - ((((data["laundry_in_building"] - (data["price"] - data["manager_id_mean_medium"])) + data["building_id_mean_medium"])/2.0) - ((data["private_terrace"] < data["price_per_room"]).astype(float))))) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["no_fee"] + ((data["manager_id_mean_high"] + (data["reduced_fee"] - (((data["price"] + ((0.836735 > ((data["site_parking_lot"] > (data["longitude"] * 2.0)).astype(float))).astype(float))) * 2.0) * 2.0)))/2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["hardwood_floors"] + (((data["building_id_mean_medium"] + (data["manager_id_mean_high"] - (((data["prewar"] + (data["prewar"] + ((data["price_per_bed"] > data["health_club"]).astype(float)))) * 2.0) * 2.0))) * 2.0) * 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((((((((-(data["price_per_room"])) > data["laundry_room"]).astype(float)) + data["manager_id_mean_medium"])/2.0) - ((data["longitude"] > (data["ft_doorman"] / 2.0)).astype(float))) * 2.0) - ((data["manager_id_mean_medium"] > (-(data["garage"]))).astype(float))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["dining_room"] + (data["furnished"] + ((data["private_outdoor_space"] * 2.0) - (data["price"] + (data["lowrise"] + ((data["private_outdoor_space"] < (data["price"] - data["num_photos"])).astype(float))))))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["num_description_words"] + (data["cats_allowed"] + ((((data["post_war"] < (-(data["latitude"]))).astype(float)) * 2.0) + (((((data["bedrooms"] * 2.0) - data["price"]) + data["no_fee"]) - data["price"]) * 2.0))))) +
        0.100000*np.tanh((data["furnished"] + (0.816667 + ((data["building_id"] + (data["reduced_fee"] - ((((data["latitude"] < data["private"]).astype(float)) * 2.0) * 2.0))) - ((data["children"] < (data["fitness"] + data["latitude"])).astype(float)))))) +
        0.100000*np.tanh((data["common_outdoor_space"] - ((data["hardwood"] + (data["dining_room"] + np.tanh(data["price_per_room"]))) - ((data["laundry_in_unit"] / 2.0) - (data["price"] - (-((data["num_photos"] * data["num_photos"])))))))) +
        0.100000*np.tanh((2.227270 + (data["price"] + ((0.836735 + (data["price"] + (((0.836735 - (data["price"] + (((data["longitude"] > data["playroom"]).astype(float)) * 2.0))) * 2.0) * 2.0))) * 2.0)))) +
        0.100000*np.tanh((-((((0.036585 < (data["latitude"] * 2.0)).astype(float)) - ((((data["bedrooms"] - ((0.036585 < (-(data["latitude"]))).astype(float))) - (data["simplex"] - data["num_photos"])) - data["price"]) * 2.0))))) +
        0.100000*np.tanh((data["furnished"] - ((data["price"] - (((data["renovated"] - (data["simplex"] + ((data["reduced_fee"] < (data["price"] + ((data["publicoutdoor"] < (-(data["doorman"]))).astype(float)))).astype(float)))) * 2.0) * 2.0)) * 2.0))) +
        0.100000*np.tanh((np.tanh(((((-(data["street_address"])) * 2.0) * 2.0) * 2.0)) - ((data["building_id_mean_medium"] * data["street_address"]) - (3.937500 * (data["laundry_in_building"] * ((data["garage"] > data["building_id_mean_medium"]).astype(float))))))) +
        0.100000*np.tanh(((((data["manager_id_mean_medium"] - ((data["longitude"] - ((data["longitude"] < ((data["manager_id"] < data["roof"]).astype(float))).astype(float))) + (((((data["no_pets"] > data["latitude"]).astype(float)) * 2.0) * 2.0) * 2.0))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((np.tanh((((data["furnished"] / 2.0) / 2.0) / 2.0)) - ((data["longitude"] > ((1.555560 < (-(data["price"]))).astype(float))).astype(float))) + ((data["bedrooms"] + (-(data["price"])))/2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh((((np.tanh((data["no_fee"] - (((data["display_address"] > ((data["no_fee"] + -1.0)/2.0)).astype(float)) * 2.0))) - (data["price"] - (data["exclusive"] - ((data["longitude"] * 2.0) * 2.0)))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["bedrooms"] > 0.563830).astype(float)) + ((data["manager_id_mean_high"] < (data["private_outdoor_space"] - data["building_id_mean_medium"])).astype(float))) - ((((data["prewar"] > ((data["num_photos"] + ((0.563830 > data["building_id_mean_medium"]).astype(float)))/2.0)).astype(float)) * 2.0) * 2.0))) +
        0.100000*np.tanh(((data["num_description_words"] * ((-(data["high_speed_internet"])) - data["laundry_in_building"])) - (((data["longitude"] * 2.0) * 2.0) - (data["common_outdoor_space"] + ((data["latitude"] < (data["fitness"] - (data["longitude"] * 2.0))).astype(float)))))) +
        0.100000*np.tanh((((((((data["manager_id_mean_medium"] * ((data["site_parking_lot"] - data["hardwood_floors"]) * 2.0)) / 2.0) + np.tanh(((-(data["manager_id_mean_medium"])) + (1.032260 - data["listing_id"])))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((((data["building_id_mean_high"] - data["building_id"]) + ((((((data["dishwasher"] * data["building_id_mean_high"]) - data["building_id"]) * 2.0) - data["roof_deck"]) * 2.0) * data["building_id_mean_high"])) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((((data["building_id_mean_high"] * (((data["manager_id_mean_high"] + ((data["prewar"] + data["manager_id_mean_medium"])/2.0))/2.0) + data["exclusive"])) - ((data["in_superintendent"] > (data["building_id"] + np.tanh(2.227270))).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((-((((data["business_center"] < ((data["longitude"] + ((data["high_ceiling"] + (((data["fitness_center"] > ((data["price"] + data["deck"])/2.0)).astype(float)) * data["deck"]))/2.0))/2.0)).astype(float)) * 2.0))) - ((data["price"] + data["prewar"])/2.0))) +
        0.100000*np.tanh((data["furnished"] + (data["manager_id_mean_high"] * ((data["manager_id_mean_high"] - (-((data["listing_id"] - ((data["exclusive"] < (data["manager_id_mean_medium"] - np.tanh(data["manager_id_mean_medium"]))).astype(float)))))) - np.tanh(np.tanh(data["manager_id_mean_medium"])))))) +
        0.100000*np.tanh((((data["cats_allowed"] + ((data["bedrooms"] + (((data["cats_allowed"] + ((data["bedrooms"] - data["price"]) - data["virtual_doorman"]))/2.0) - data["washer_in_unit"])) + (data["bedrooms"] - data["price"]))) * 2.0) * 2.0)) +
        0.100000*np.tanh((((-(data["in_superintendent"])) - data["high_speed_internet"]) - ((data["new_construction"] + (((((data["view"] > ((data["num_photos"] * ((data["num_photos"] * data["live_in_super"]) / 2.0)) / 2.0)).astype(float)) * 2.0) * 2.0) * 2.0))/2.0))) +
        0.100000*np.tanh((-(((data["price"] - ((0.563830 - (((data["longitude"] > (data["longitude"] / 2.0)).astype(float)) * 2.0)) - (data["listing_id"] * data["listing_id"]))) - (((data["bathrooms"] > 0.906250).astype(float)) * 2.0))))) +
        0.100000*np.tanh((-((data["simplex"] - ((-((data["longitude"] - (-(((((data["washer_in_unit"] - data["price_per_bath"]) > ((((-(data["bathrooms"])) < np.tanh(data["price_per_bath"])).astype(float)) * 2.0)).astype(float)) * 2.0)))))) * 2.0))))) +
        0.100000*np.tanh(((((((-((0.836735 + data["price"]))) * ((data["latitude"] < data["_photos"]).astype(float))) * 2.0) * 2.0) - (data["num_description_words"] * ((data["new_construction"] + data["simplex"]) + data["stainless_steel_appliances"]))) * 2.0)) +
        0.100000*np.tanh(((data["renovated"] + (data["prewar"] * (data["building_id_mean_medium"] * 2.0))) + (data["pre"] * (data["building_id_mean_medium"] + ((-(((-(data["war"])) - (data["building_id_mean_medium"] * 2.0)))) * 2.0))))) +
        0.100000*np.tanh((((data["furnished"] + ((data["reduced_fee"] + data["bedrooms"])/2.0))/2.0) + (-((data["price"] + (((data["reduced_fee"] < ((data["war"] + ((data["price_per_bath"] < ((data["war"] + data["price_per_bed"])/2.0)).astype(float)))/2.0)).astype(float)) * 2.0)))))) +
        0.100000*np.tanh(((((data["common_terrace"] > (data["longitude"] * data["bathrooms"])).astype(float)) - (((data["longitude"] > (data["latitude"] + (data["bathrooms"] * (-(data["common_terrace"]))))).astype(float)) * 2.0)) + ((data["_photos"] > data["longitude"]).astype(float)))) +
        0.100000*np.tanh((data["bedrooms"] * (-(((data["cats_allowed"] - (-((data["private_outdoor_space"] - (-(((data["new_construction"] + ((data["no_fee"] + ((data["no_fee"] + data["laundry_in_building"])/2.0))/2.0)) * 2.0))))))) * 2.0))))) +
        0.100000*np.tanh(((((((((data["manager_id_mean_medium"] * (data["view"] * 2.0)) + ((data["manager_id_mean_medium"] + data["high_ceilings"]) + data["level"])) / 2.0) > np.tanh(np.tanh(data["manager_id_mean_medium"]))).astype(float)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((((data["prewar"] - (data["manager_id_mean_medium"] / 2.0)) > data["building_id_mean_high"]).astype(float)) + (data["building_id_mean_high"] * ((5.67775487899780273) * ((5.67775487899780273) * ((data["prewar"] > ((data["site_garage"] + (data["manager_id_mean_medium"] / 2.0))/2.0)).astype(float))))))) +
        0.099980*np.tanh((data["price_per_bath"] - (((((data["price"] + ((data["duplex"] < ((data["exclusive"] * (data["green_building"] * (data["building_id_mean_high"] * 2.0))) * 2.0)).astype(float)))/2.0) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh(((data["laundry_in_unit"] + ((data["longitude"] < ((data["gym_in_building"] - (data["laundry"] * ((data["longitude"] > (-((data["indoor_pool"] * (0.836735 * data["price_per_room"]))))).astype(float)))) - data["latitude"])).astype(float))) * 2.0)) +
        0.100000*np.tanh(((data["new_construction"] + ((data["private_outdoor_space"] + ((data["furnished"] > ((data["manager_id_mean_high"] * (data["display_address"] - (data["renovated"] + ((data["no_fee"] * data["manager_id_mean_high"]) / 2.0)))) / 2.0)).astype(float))) * 2.0)) * 2.0)) +
        0.100000*np.tanh(((((((-(((data["private"] > data["latitude"]).astype(float)))) * 2.0) + ((((-(data["latitude"])) - ((data["price_per_room"] > (data["live_in_super"] / 2.0)).astype(float))) > (data["wifi_access"] / 2.0)).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["parking_space"] + (((data["new_construction"] > (data["price_per_room"] * ((data["site_parking_lot"] > data["longitude"]).astype(float)))).astype(float)) - ((data["price"] + (data["concierge"] + (-((data["street_address"] * (-(data["fitness_center"])))))))/2.0)))) +
        0.100000*np.tanh(((data["balcony"] + ((data["bedrooms"] > ((data["display_address"] + (4.250000 / 2.0))/2.0)).astype(float))) + (data["bathrooms"] + (((data["bedrooms"] > ((data["bathrooms"] + (4.250000 / 2.0))/2.0)).astype(float)) - data["price"])))) +
        0.100000*np.tanh((-((((data["outdoor_areas"] > ((data["longitude"] + (data["_pets_ok_"] * data["roof_deck"])) - ((data["outdoor_areas"] > (data["roofdeck"] - ((data["longitude"] + (data["speed_internet"] * data["outdoor_space"]))/2.0))).astype(float)))).astype(float)) * 2.0)))) +
        0.100000*np.tanh(((data["building_id_mean_medium"] * data["simplex"]) + (data["num_photos"] + np.tanh(((((((1.555560 - ((data["listing_id"] + (data["bathrooms"] + data["listing_id"]))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))))) +
        0.099730*np.tanh((data["bedrooms"] + ((data["wheelchair_access"] - (0.906250 - (data["price_per_bath"] + (4.250000 * (-(((data["price"] + (((data["price"] / 2.0) > data["no_fee"]).astype(float)))/2.0))))))) * 2.0))) +
        0.100000*np.tanh((((((((data["longitude"] < data["private_balcony"]).astype(float)) - ((data["price"] + (-(((data["doorman"] + ((data["longitude"] < (data["ft_doorman"] - data["_dishwasher_"])).astype(float))) * 2.0))))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["bedrooms"] - ((((data["listing_id"] + data["price"])/2.0) + data["price"])/2.0)) - ((data["private"] > ((data["listing_id"] + ((data["latitude"] > data["longitude"]).astype(float)))/2.0)).astype(float))) - ((data["central_a"] > data["latitude"]).astype(float)))) +
        0.100000*np.tanh((data["manager_id_mean_medium"] * ((((((-(((data["concierge"] < ((((data["building_id_mean_medium"] / 2.0) / 2.0) / 2.0) / 2.0)).astype(float)))) + 0.604651) * 2.0) + (data["manager_id_mean_medium"] / 2.0)) * 2.0) * 2.0))) +
        0.100000*np.tanh(((((data["parking_space"] + data["multi"]) + data["reduced_fee"]) + (data["private_outdoor_space"] - (data["price"] - (((data["bathrooms"] + (-(data["dining_room"]))) + (data["hardwood"] * data["building_id"]))/2.0))))/2.0)) +
        0.100000*np.tanh((data["listing_id"] * (data["building_id_mean_high"] - ((((data["manager_id"] - (data["building_id_mean_high"] - ((data["reduced_fee"] - data["dishwasher"]) + data["price"]))) + data["renovated"]) + data["fireplace"]) + data["price"])))) +
        0.100000*np.tanh((data["new_construction"] * ((data["subway"] + (data["war"] + ((((data["longitude"] < data["valet"]).astype(float)) - data["street_address"]) - data["terrace"]))) + (((data["longitude"] < data["subway"]).astype(float)) - data["display_address"])))) +
        0.100000*np.tanh(((((data["reduced_fee"] - (data["reduced_fee"] * 2.0)) - ((((data["_dryer"] > (data["longitude"] * ((-(1.361700)) * 2.0))).astype(float)) > np.tanh(((1.361700 + data["num_photos"])/2.0))).astype(float))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((-(((data["decorative_fireplace"] > ((data["latitude"] - (((data["price_per_room"] > 1.860470).astype(float)) + data["granite_kitchen"])) / 2.0)).astype(float)))) - ((data["price"] + (((data["latitude"] + data["latitude"]) < data["street_address"]).astype(float)))/2.0))) +
        0.100000*np.tanh((((((0.604651 + data["price"])/2.0) > 0.437500).astype(float)) - ((data["balcony"] * data["laundry_in_building"]) + ((((0.437500 > data["price"]).astype(float)) + (data["in_super"] + (data["highrise"] + data["price"])))/2.0)))) +
        0.100000*np.tanh((((data["common_terrace"] > (data["highrise"] * data["listing_id"])).astype(float)) - ((((((1.361700 + data["num_photos"])/2.0) < ((((data["highrise"] > (data["decorative_fireplace"] * data["listing_id"])).astype(float)) * 2.0) * 2.0)).astype(float)) * 2.0) * 2.0))) +
        0.100000*np.tanh(((data["loft"] * (data["elevator"] - (data["building_id_mean_medium"] + (data["actual_apt"] + (data["storage"] + ((data["price_per_room"] + data["short_term_allowed"])/2.0)))))) - ((data["building_id_mean_medium"] > ((data["price_per_room"] < 1.860470).astype(float))).astype(float)))) +
        0.100000*np.tanh(((data["actual_apt"] - ((0.604651 < (-(((data["num_photos"] + data["subway"])/2.0)))).astype(float))) - ((((((data["listing_id"] + data["num_photos"])/2.0) + data["outdoor_areas"])/2.0) + data["site_garage"]) + data["subway"]))) +
        0.100000*np.tanh(((-((((data["storage"] + ((data["price_per_room"] < 1.916670).astype(float))) < (-(data["building_id_mean_medium"]))).astype(float)))) + (-((data["residents_lounge"] - ((data["building_id_mean_medium"] + data["patio"]) * (-(data["street_address"])))))))) +
        0.092170*np.tanh((-((data["building_id"] * ((((data["building_id_mean_medium"] - np.tanh(data["num_description_words"])) - data["dogs_allowed"]) - (-(data["common_roof_deck"]))) - (data["hardwood"] + np.tanh(data["num_description_words"]))))))) +
        0.100000*np.tanh((((data["price_per_bath"] + (-(((data["listing_id"] > (0.836735 * 2.0)).astype(float)))))/2.0) * ((data["garage"] + (((data["bathrooms"] > (0.836735 * 2.0)).astype(float)) + ((data["bathrooms"] < 0.836735).astype(float)))) * 2.0))) +
        0.100000*np.tanh(((data["building_id"] + ((-((data["fitness_center"] * (-((data["high_speed_internet"] + (data["level"] + ((data["listing_id"] - data["balcony"]) - data["green_building"])))))))) + (data["bathrooms"] - (0.76526659727096558))))/2.0)) +
        0.100000*np.tanh((-(((data["laundry_in_unit"] > ((data["furnished"] + ((data["level"] + (-(((((data["laundry_in_unit"] + ((data["site_garage"] + data["num_photos"])/2.0))/2.0) * (data["num_photos"] * 2.0)) + data["washer_in_unit"]))))/2.0))/2.0)).astype(float))))) +
        0.100000*np.tanh((((data["war"] * (-(data["laundry_in_unit"]))) + (((data["no_fee"] * ((data["building_id_mean_medium"] - data["laundry_in_building"]) - data["laundry"])) + np.tanh(((-(data["laundry_in_unit"])) * 2.0))) / 2.0)) * 2.0)) +
        0.100000*np.tanh(((-((data["simplex"] * 2.0))) + (((-(((data["simplex"] > (data["common_roof_deck"] + (data["decorative_fireplace"] - (data["latitude"] + ((data["roof"] > data["latitude"]).astype(float)))))).astype(float)))) * 2.0) * 2.0))) +
        0.095440*np.tanh((-((data["no_fee"] * (np.tanh(((((data["doorman"] * (data["hardwood_floors"] + (data["new_construction"] - ((data["patio"] > (0.036585 - data["price"])).astype(float))))) * 2.0) * 2.0) * 2.0)) * 2.0))))) +
        0.100000*np.tanh(((-(((0.836735 < ((data["listing_id"] + data["_pets_ok_"])/2.0)).astype(float)))) - (((((data["shares_ok"] > ((data["latitude"] * data["no_fee"]) * data["price_per_bath"])).astype(float)) * 2.0) * 2.0) * 2.0))) +
        0.082700*np.tanh(((((data["exclusive"] - (data["latitude"] - data["exclusive"])) + ((data["bedrooms"] * (np.tanh(2.400000) / 2.0)) + ((data["eat_in_kitchen"] > (-(data["latitude"]))).astype(float)))) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["price_per_bath"] + ((data["doorman"] * (((data["outdoor_space"] < data["price_per_room"]).astype(float)) + ((data["site_garage"] < data["laundry_in_unit"]).astype(float)))) * 2.0))/2.0) + (((data["latitude"] + np.tanh(data["longitude"])) < data["site_super"]).astype(float)))) +
        0.100000*np.tanh((data["short_term_allowed"] + (((data["wheelchair_access"] * ((data["multi"] - (data["24"] * 2.0)) - ((data["dishwasher"] * 2.0) * 2.0))) + (-((data["price"] + ((data["price"] > data["wheelchair_access"]).astype(float))))))/2.0))) +
        0.100000*np.tanh((((((-((data["building_id_mean_medium"] - (-(data["manager_id_mean_high"]))))) > data["manager_id_mean_medium"]).astype(float)) + ((data["longitude"] > ((data["manager_id_mean_medium"] > data["building_id_mean_high"]).astype(float))).astype(float))) - ((data["longitude"] > (data["longitude"] * data["building_id_mean_high"])).astype(float)))) +
        0.100000*np.tanh((((((data["bathrooms"] > (data["fireplace"] - (data["stainless_steel_appliances"] - 0.836735))).astype(float)) * 2.0) * 2.0) + (((data["price_per_bath"] + ((data["private_outdoor_space"] * 2.0) * 2.0)) + data["high_ceiling"]) + data["private_outdoor_space"]))) +
        0.097140*np.tanh((data["num_photos"] * (-((data["work"] + (data["num_photos"] + (data["reduced_fee"] + ((-((((data["cats_allowed"] + data["price_per_bath"]) + (data["cats_allowed"] + data["central_a"]))/2.0))) * 2.0)))))))) +
        0.092690*np.tanh((((((((((data["hardwood_floors"] * (((data["building_id_mean_medium"] * data["manager_id_mean_high"]) * 2.0) * 2.0)) + (-(((data["manager_id_mean_high"] + data["common_outdoor_space"])/2.0))))/2.0) - data["latitude"]) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["building_id_mean_high"] * (data["dogs_allowed"] + (-((0.816667 + (data["price"] + (data["manager_id"] - (((((data["price"] > ((data["cats_allowed"] > data["building_id_mean_high"]).astype(float))).astype(float)) * 2.0) * 2.0) * 2.0))))))))) +
        0.100000*np.tanh(((0.836735 < (data["indoor_pool"] - ((data["price"] - ((((data["latitude"] < (data["price_per_bed"] * ((data["high_speed_internet"] - data["private_outdoor_space"]) - data["price"]))).astype(float)) < data["price_per_bath"]).astype(float))) / 2.0))).astype(float))) +
        0.100000*np.tanh((((data["_photos"] + (((((data["common_outdoor_space"] + (data["hardwood_floors"] * data["display_address"])) < (((data["hardwood_floors"] * data["laundry_in_building"]) < data["display_address"]).astype(float))).astype(float)) * (data["display_address"] * data["laundry_in_building"])) * 2.0))/2.0) * 2.0)) +
        0.100000*np.tanh((((data["dishwasher"] * ((data["terrace"] > data["doorman"]).astype(float))) * data["terrace"]) - (data["price_per_bed"] * (data["high_speed_internet"] - (data["laundry_in_building"] + ((data["storage"] < ((data["num_description_words"] / 2.0) / 2.0)).astype(float))))))) +
        0.100000*np.tanh((data["high_ceiling"] + ((data["furnished"] + (data["latitude"] + ((data["actual_apt"] > (((data["latitude"] / 2.0) + ((0.836735 + (data["building_id_mean_high"] + data["dryer_in_unit"]))/2.0)) * 2.0)).astype(float)))) * 2.0))) +
        0.100000*np.tanh((((data["high_ceilings"] + ((data["residents_lounge"] > (-((((data["street_address"] > (np.tanh(0.836735) - -1.0)).astype(float)) - np.tanh(np.tanh((data["manager_id_mean_high"] * data["manager_id_mean_medium"]))))))).astype(float))) * 2.0) * 2.0)) +
        0.100000*np.tanh((((-((((data["longitude"] > (-(((data["in_super"] * data["price"]) * data["price"])))).astype(float)) - (((data["price"] * data["highrise"]) * data["price"]) * data["price"])))) * 2.0) * 2.0)) +
        0.099990*np.tanh((data["manager_id_mean_high"] - (-((data["bathrooms"] - (((((((((-(-1.0)) < data["bathrooms"]).astype(float)) * 2.0) < data["bathrooms"]).astype(float)) * 2.0) * 2.0) * 2.0)))))) +
        0.100000*np.tanh((-(((((((data["latitude"] < ((((data["washer_in_unit"] < (data["latitude"] + data["simplex"])).astype(float)) + (data["simplex"] + data["site_garage"]))/2.0)).astype(float)) * 2.0) * 2.0) * 2.0) - ((data["price_per_bath"] > data["manager_id_mean_medium"]).astype(float)))))) +
        0.100000*np.tanh((-(((((data["listing_id"] > 1.916670).astype(float)) + (-((data["swimming_pool"] * (data["swimming_pool"] * (data["microwave"] + (data["high_ceiling"] + (-((data["doorman"] * data["num_description_words"])))))))))) * 2.0)))) +
        0.100000*np.tanh((data["parking_space"] - (-(((data["indoor_pool"] > (-(((data["simplex"] > np.tanh((data["new_construction"] * ((data["street_address"] + (data["manager_id"] - (data["laundry"] * 2.0)))/2.0)))).astype(float))))).astype(float)))))) +
        0.100000*np.tanh((((data["lowrise"] - (data["site_garage"] - (data["display_address"] * ((data["building_id"] < (data["private_outdoor_space"] - ((data["manager_id_mean_medium"] > ((data["site_laundry"] + data["building_id"])/2.0)).astype(float)))).astype(float))))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["newly_renovated"] > (data["price_per_bed"] * ((data["manager_id_mean_medium"] > ((data["garden"] - data["price_per_bed"]) + ((data["dogs_allowed"] + ((data["garden"] - (data["dining_room"] * 2.0)) + (data["laundry_in_building"] * 2.0)))/2.0))).astype(float)))).astype(float))) +
        0.100000*np.tanh(((data["price_per_bed"] * (data["private_outdoor_space"] + ((data["pre"] > (data["building_id_mean_medium"] / 2.0)).astype(float)))) - ((1.246580 < (data["outdoor_areas"] + (data["private_outdoor_space"] + (np.tanh((-(data["building_id_mean_medium"]))) * 2.0)))).astype(float)))) +
        0.100000*np.tanh((data["wheelchair_access"] * ((np.tanh(((((((data["street_address"] * (0.836735 + ((data["dogs_allowed"] - data["longitude"]) - data["longitude"]))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) * 2.0) * 2.0))))
    return Outputs(p) 


def GPNotLo3(data):
    p = (-0.822105 +
        0.100000*np.tanh((((data["building_id"] + (data["hardwood_floors"] + data["laundry_in_building"])) + ((data["bedrooms"] + (data["building_id_mean_high"] + ((data["manager_id_mean_high"] + (data["reduced_fee"] - data["price"])) * 2.0))) * 2.0)) * 2.0)) +
        0.100000*np.tanh(((14.78523731231689453) * ((((data["latitude"] < ((data["garden"] < data["no_fee"]).astype(float))).astype(float)) + ((data["building_id_mean_high"] + (data["building_id_mean_medium"] + ((data["manager_id_mean_medium"] + data["manager_id_mean_high"]) * 2.0))) * 2.0)) + data["no_fee"]))) +
        0.100000*np.tanh(((-1.0 + ((data["manager_id_mean_medium"] + (data["manager_id_mean_high"] - (data["price"] - (((data["bedrooms"] - (data["price"] - data["building_id_mean_high"])) + data["no_fee"]) + data["manager_id_mean_medium"])))) * 2.0)) * 2.0)) +
        0.100000*np.tanh(((((((data["no_fee"] + ((data["manager_id_mean_high"] - data["price"]) * 2.0)) + (np.tanh(np.tanh((data["building_id_mean_high"] * 2.0))) * 2.0)) * 2.0) + data["price"]) + data["dishwasher"]) * 2.0)) +
        0.100000*np.tanh((((((data["reduced_fee"] - (data["doorman"] * 2.0)) + (((data["manager_id_mean_high"] + (data["building_id_mean_medium"] + (data["manager_id_mean_medium"] * 2.0))) + (data["building_id_mean_high"] + data["manager_id_mean_medium"])) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["no_fee"] + ((((np.tanh(data["building_id_mean_medium"]) + (data["manager_id_mean_medium"] + ((data["bedrooms"] + np.tanh((data["reduced_fee"] * 2.0))) - data["price"]))) * 2.0) - data["price"]) * 2.0)) * 2.0)) +
        0.100000*np.tanh((data["no_fee"] + (data["hardwood_floors"] + (data["building_id"] + (5.785710 * ((((data["manager_id_mean_high"] - ((data["price_per_bath"] < data["laundry"]).astype(float))) * 2.0) + ((data["price_per_bath"] < (0.843137 / 2.0)).astype(float)))/2.0)))))) +
        0.100000*np.tanh((((data["manager_id_mean_high"] + ((data["building_id_mean_medium"] + ((data["manager_id_mean_medium"] - ((data["price"] * 2.0) + np.tanh(2.282050))) + data["bedrooms"])) * 2.0)) + (data["no_fee"] + data["furnished"])) * 2.0)) +
        0.100000*np.tanh((data["manager_id_mean_high"] + (data["bedrooms"] + ((data["laundry_in_unit"] + (data["building_id_mean_medium"] + ((((data["manager_id_mean_medium"] + (data["furnished"] - ((data["lowrise"] > data["bedrooms"]).astype(float))))/2.0) - data["price"]) * 2.0))) * 2.0)))) +
        0.100000*np.tanh((data["reduced_fee"] + ((((data["laundry_in_building"] + data["manager_id_mean_medium"]) + (data["parking_space"] + (data["building_id_mean_medium"] + (-((data["price"] - data["bedrooms"])))))) + (-(data["price"]))) * 2.0))) +
        0.100000*np.tanh((((((((data["reduced_fee"] - data["listing_id"]) + data["building_id_mean_medium"])/2.0) + (data["manager_id_mean_high"] - 1.174600)) + ((data["building_id_mean_high"] + (data["manager_id_mean_medium"] - data["price"])) * 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["furnished"] + (((data["building_id_mean_medium"] + data["no_fee"]) + (data["common_outdoor_space"] + (data["manager_id_mean_high"] - (np.tanh(((((data["price"] + 0.843137)/2.0) * 2.0) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
        0.100000*np.tanh(((((((((data["manager_id_mean_medium"] + np.tanh(((data["garage"] < (data["building_id_mean_medium"] * 2.0)).astype(float)))) * 2.0) * 2.0) + np.tanh(np.tanh(((data["building_id_mean_high"] * 2.0) * 2.0)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((8.0) * (((data["reduced_fee"] - data["building_id_mean_medium"]) + ((6.0) * ((data["laundry_in_unit"] + ((data["building_id_mean_medium"] + (data["short_term_allowed"] + (data["building_id_mean_high"] * 2.0))) * 2.0))/2.0))) * 2.0)) * 2.0)) +
        0.100000*np.tanh((((((((data["manager_id_mean_medium"] - (data["price"] - data["bedrooms"])) - (data["hardwood"] - data["bedrooms"])) - data["price"]) + (data["no_fee"] - data["listing_id"])) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["manager_id_mean_high"] + ((data["laundry_in_building"] + ((data["reduced_fee"] - (data["price"] + ((data["marble_bath"] < ((data["price_per_bed"] * ((data["simplex"] < data["bedrooms"]).astype(float))) * 2.0)).astype(float)))) * 2.0)) * 2.0))) +
        0.100000*np.tanh(((data["reduced_fee"] + ((data["manager_id_mean_high"] - (((data["furnished"] < (((data["price"] / 2.0) - data["dining_room"]) / 2.0)).astype(float)) * 2.0)) + data["furnished"])) - (data["price"] - data["manager_id_mean_medium"]))) +
        0.100000*np.tanh(((data["hardwood_floors"] - (data["price"] - (data["private_outdoor_space"] + np.tanh(((data["num_photos"] + np.tanh(((1.174600 - (data["doorman"] * (data["building_id_mean_medium"] * 2.0))) * 2.0))) * 2.0))))) * 2.0)) +
        0.100000*np.tanh(((((data["building_id_mean_high"] + data["furnished"])/2.0) + ((data["common_outdoor_space"] + (-(data["price"]))) + (np.tanh(data["num_photos"]) / 2.0))) + (data["bathrooms"] - ((data["longitude"] > (data["children"] / 2.0)).astype(float))))) +
        0.100000*np.tanh((((data["bedrooms"] - ((-((((-1.0 > data["price"]).astype(float)) - (((data["washer_in_unit"] > (data["childrens_playroom"] - (data["longitude"] - data["sauna"]))).astype(float)) * 2.0)))) + data["price"])) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["balcony"] + (data["dishwasher"] + (((data["building_id_mean_medium"] + ((data["longitude"] < data["private_terrace"]).astype(float))) + (data["laundry_in_building"] + (((data["longitude"] < data["hi_rise"]).astype(float)) - data["price"]))) * 2.0))) * 2.0)) +
        0.100000*np.tanh(((data["furnished"] + (data["private_outdoor_space"] + ((data["no_fee"] * data["no_fee"]) + (((data["bedrooms"] - (data["price"] - (-(((data["24"] < data["price_per_bed"]).astype(float)))))) * 2.0) * 2.0)))) * 2.0)) +
        0.100000*np.tanh(((data["renovated"] - (data["hardwood"] - ((((data["no_fee"] + (((data["private_outdoor_space"] + data["reduced_fee"]) + data["cats_allowed"])/2.0))/2.0) - (data["price"] + ((data["new_construction"] > data["num_photos"]).astype(float)))) * 2.0))) * 2.0)) +
        0.100000*np.tanh((((-1.0 - (-((data["num_photos"] - (data["high_speed_internet"] + (data["live"] + (data["virtual_doorman"] + ((data["simplex"] - ((5.888890 + data["manager_id_mean_medium"])/2.0)) / 2.0)))))))) * 2.0) * 2.0)) +
        0.100000*np.tanh(((7.0) * (((6.0) * ((-(((-((((data["gym_in_building"] - data["wifi_access"]) > data["latitude"]).astype(float)))) + ((data["private"] > data["latitude"]).astype(float))))) + data["renovated"])) + data["furnished"]))) +
        0.100000*np.tanh((((data["wheelchair_access"] + (((data["laundry_in_unit"] - (data["price"] + (((-(((data["longitude"] < ((data["sauna"] / 2.0) / 2.0)).astype(float)))) > ((data["new_construction"] + data["wheelchair_access"])/2.0)).astype(float)))) * 2.0) * 2.0))/2.0) * 2.0)) +
        0.100000*np.tanh((data["no_fee"] + (((data["price_per_room"] + (4.333330 * (-(((((data["longitude"] > (-(((data["view"] > data["latitude"]).astype(float))))).astype(float)) + np.tanh(data["price_per_room"]))/2.0))))) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["bedrooms"] - (data["price"] + (((((data["loft"] < (-(((data["virtual_doorman"] < ((0.843137 + ((data["manager_id_mean_medium"] - data["dining_room"]) - data["dryer"]))/2.0)).astype(float))))).astype(float)) * 2.0) * 2.0) * 2.0)))) +
        0.100000*np.tanh((((data["street_address"] < data["bathrooms"]).astype(float)) - (data["price"] - (data["num_description_words"] - (data["building_id_mean_medium"] * ((data["manager_id_mean_medium"] + data["building_id_mean_medium"]) + ((data["building_id"] + 1.0) + data["street_address"]))))))) +
        0.100000*np.tanh(((data["dogs_allowed"] - ((data["num_photos"] - np.tanh(data["bedrooms"])) - 1.0)) + (((((-(((data["simplex"] > (data["num_photos"] + 1.0)).astype(float)))) * 2.0) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh((data["no_fee"] - ((((data["longitude"] + (data["price"] + ((data["longitude"] > (-(((data["exclusive"] < (((data["exclusive"] + data["building_id_mean_high"])/2.0) / 2.0)).astype(float))))).astype(float)))) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh(((((data["num_photos"] + ((data["actual_apt"] > (data["indoor_pool"] + data["longitude"])).astype(float))) - ((((data["longitude"] > ((data["wheelchair_ramp"] + ((data["bedrooms"] < data["green_building"]).astype(float)))/2.0)).astype(float)) * 2.0) * 2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["bedrooms"] - ((data["brownstone"] < data["street_address"]).astype(float))) - (data["price"] - ((((data["price_per_bed"] * data["no_pets"]) > data["latitude"]).astype(float)) + ((-(((data["latitude"] < data["no_pets"]).astype(float)))) * 2.0))))) +
        0.100000*np.tanh(((data["reduced_fee"] + ((data["exclusive"] > data["manager_id_mean_medium"]).astype(float))) + (5.888890 * (5.888890 * (data["building_id_mean_high"] * ((data["manager_id_mean_medium"] < (data["simplex"] - 0.616438)).astype(float))))))) +
        0.100000*np.tanh((((data["longitude"] < data["flex"]).astype(float)) + ((data["num_description_words"] * (1.0 - (np.tanh(np.tanh(((data["laundry_in_building"] + (((data["price_per_room"] < (data["lowrise"] / 2.0)).astype(float)) * 2.0))/2.0))) * 2.0))) * 2.0))) +
        0.100000*np.tanh(((-(data["price"])) + ((data["furnished"] + np.tanh(((data["exclusive"] - (data["building_id_mean_high"] * (data["multi"] + (((-((data["hardwood"] * 2.0))) > data["manager_id_mean_high"]).astype(float))))) * 2.0))) * 2.0))) +
        0.100000*np.tanh((data["manager_id_mean_high"] + (((data["bedrooms"] / 2.0) - ((data["fireplace"] < data["24"]).astype(float))) + (((((data["wheelchair_access"] > ((((data["manager_id_mean_medium"] < -1.0).astype(float)) + data["manager_id_mean_high"])/2.0)).astype(float)) * 2.0) * 2.0) * 2.0)))) +
        0.100000*np.tanh(((((data["price_per_bed"] < data["site_parking_lot"]).astype(float)) - ((data["longitude"] > (0.692308 * (data["price"] * data["site_parking_lot"]))).astype(float))) - ((data["longitude"] > (0.692308 * (data["price"] * data["site_parking_lot"]))).astype(float)))) +
        0.100000*np.tanh((((data["_photos"] > ((((data["latitude"] + data["longitude"])/2.0) * 2.0) * 2.0)).astype(float)) + (data["furnished"] - ((((((data["latitude"] + ((data["no_pets"] + data["eat_in_kitchen"])/2.0))/2.0) < data["no_pets"]).astype(float)) * 2.0) * 2.0)))) +
        0.100000*np.tanh(((((data["building_id"] - (data["price"] + data["price_per_room"])) - ((data["hardwood_floors"] + ((data["building_id_mean_medium"] * (data["street_address"] + (((data["hardwood_floors"] * 2.0) + data["price_per_room"])/2.0))) * 2.0))/2.0)) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["bathrooms"] + ((data["bathrooms"] + (((((data["price"] < ((data["balcony"] - data["hardwood"]) - ((data["patio"] > data["doorman"]).astype(float)))).astype(float)) + (data["common_outdoor_space"] * 2.0)) * 2.0) * 2.0)) * 2.0))) +
        0.100000*np.tanh((((((((data["reduced_fee"] + (data["parking_space"] + (np.tanh(((data["price_per_bath"] + (data["war"] * data["building_id"]))/2.0)) * 2.0)))/2.0) + (data["hardwood"] * data["building_id_mean_medium"])) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.099980*np.tanh((np.tanh((((((((data["no_fee"] / 2.0) / 2.0) - ((data["no_fee"] * data["bedrooms"]) / 2.0)) - (data["num_description_words"] * data["new_construction"])) * 2.0) / 2.0) * 2.0)) * 2.0)) +
        0.100000*np.tanh((((data["laundry_in_building"] * (((data["outdoor_space"] + data["laundry_in_building"]) + data["dogs_allowed"])/2.0)) * (data["dogs_allowed"] * (data["doorman"] + data["price_per_bed"]))) - ((data["longitude"] > (data["price"] * data["pet_friendly"])).astype(float)))) +
        0.100000*np.tanh((np.tanh(-1.0) + ((data["laundry_room"] < ((data["virtual_doorman"] * (data["price"] + ((1.174600 + (data["lowrise"] - ((data["price_per_room"] < np.tanh(-1.0)).astype(float)))) * 2.0))) * 2.0)).astype(float)))) +
        0.100000*np.tanh(((data["bathrooms"] * (data["bedrooms"] * data["doorman"])) - ((data["pool"] + (-((data["renovated"] + (data["no_fee"] * ((data["bedrooms"] - data["building_id_mean_medium"]) * data["doorman"]))))))/2.0))) +
        0.100000*np.tanh((((data["furnished"] + ((data["post_war"] < data["latitude"]).astype(float))) - ((-(data["num_photos"])) * (-(((data["num_photos"] / 2.0) - ((data["hardwood_floors"] > ((data["post_war"] > data["hardwood_floors"]).astype(float))).astype(float))))))) * 2.0)) +
        0.100000*np.tanh(((((-((data["in_super"] - (-(((data["fireplace"] > (data["listing_id"] * (-(((((data["private_backyard"] + (data["in_super"] + data["listing_id"]))/2.0) / 2.0) / 2.0))))).astype(float))))))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((-((((data["price"] + ((data["speed_internet"] < (data["prewar"] + (-((data["furnished"] - ((data["building_id_mean_high"] + data["manager_id_mean_medium"])/2.0)))))).astype(float))) * 2.0) * 2.0))) + data["no_fee"]) + data["bedrooms"])) +
        0.100000*np.tanh(((((data["short_term_allowed"] + ((data["light"] > (((data["longitude"] + ((data["childrens_playroom"] + (((data["latitude"] + (data["common_outdoor_space"] * 2.0))/2.0) * 2.0))/2.0))/2.0) / 2.0)).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((((((-(data["longitude"])) - (data["manager_id_mean_medium"] * (np.tanh(data["manager_id_mean_high"]) + (((data["building_id_mean_high"] + data["manager_id_mean_medium"])/2.0) * (-(data["manager_id_mean_high"])))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((((((data["actual_apt"] + data["private_outdoor_space"])/2.0) > (data["bathrooms"] * (((data["private_outdoor_space"] > (data["bathrooms"] * data["high_ceiling"])).astype(float)) + ((data["site_laundry"] + data["high_ceiling"])/2.0)))).astype(float)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["parking_space"] + (data["price_per_bath"] + (data["exclusive"] + (data["furnished"] + (data["actual_apt"] + (data["new_construction"] * ((data["price_per_room"] * ((data["price_per_bath"] > (-(data["parking_space"]))).astype(float))) * 2.0)))))))) +
        0.100000*np.tanh((((0.600000 < data["bathrooms"]).astype(float)) - (((-(((data["sublet"] > (data["latitude"] - (((((data["outdoor_entertainment_space"] + data["latitude"])/2.0) + data["private_terrace"])/2.0) / 2.0))).astype(float)))) < ((data["latitude"] + data["laundry"])/2.0)).astype(float)))) +
        0.100000*np.tanh((((((data["building"] < data["manager_id_mean_medium"]).astype(float)) < ((data["building_id_mean_high"] < np.tanh(np.tanh(data["elevator"]))).astype(float))).astype(float)) - ((data["manager_id_mean_medium"] * data["green_building"]) + ((data["price"] + (data["num_photos"] * data["num_photos"]))/2.0)))) +
        0.100000*np.tanh((((data["manager_id_mean_high"] < (-(data["building_id_mean_medium"]))).astype(float)) - (((((data["building_id_mean_high"] * (data["building_id_mean_medium"] * 2.0)) + (data["manager_id_mean_high"] * (data["building_id"] * 2.0)))/2.0) + (data["building_id_mean_medium"] * (data["building_id"] * 2.0)))/2.0))) +
        0.100000*np.tanh((((data["bedrooms"] + data["building_id"])/2.0) + ((-((data["dogs_allowed"] * ((data["street_address"] + data["bedrooms"])/2.0)))) - (data["street_address"] * ((data["fitness_center"] + (data["bedrooms"] + data["building_id"]))/2.0))))) +
        0.100000*np.tanh((((data["building_id_mean_medium"] > np.tanh(data["manager_id_mean_medium"])).astype(float)) - ((np.tanh(data["manager_id_mean_medium"]) + (data["building_id_mean_medium"] + data["building_id_mean_medium"])) - (data["manager_id_mean_medium"] * (data["manager_id_mean_medium"] * (data["building_id_mean_medium"] + data["building_id_mean_medium"])))))) +
        0.100000*np.tanh(((((data["hardwood_floors"] < (data["manager_id_mean_medium"] - ((data["manager_id_mean_high"] > ((-1.0 + data["air_conditioning"])/2.0)).astype(float)))).astype(float)) - ((data["latitude"] < ((data["longitude"] + data["view"])/2.0)).astype(float))) - ((data["building_id_mean_medium"] < -1.0).astype(float)))) +
        0.100000*np.tanh(((((((data["manager_id_mean_high"] > (4.333330 / 2.0)).astype(float)) - ((data["longitude"] > (data["fitness_center"] * ((data["washer_in_unit"] > (-(((data["fitness_center"] + data["roof_deck"])/2.0)))).astype(float)))).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["manager_id_mean_high"] + data["listing_id"]) * ((data["manager_id_mean_high"] + data["price_per_bath"])/2.0)) + (-((data["subway"] - (-((data["dryer_in_unit"] - (data["listing_id"] * (data["manager_id_mean_high"] + data["building_id_mean_medium"])))))))))) +
        0.100000*np.tanh(((data["level"] * ((data["listing_id"] - (((-((data["renovated"] - (-(data["parking_space"]))))) * 2.0) * 2.0)) - data["level"])) * (data["listing_id"] - ((-(data["reduced_fee"])) * 2.0)))) +
        0.100000*np.tanh((-((data["price"] + ((data["new_construction"] < (data["dogs_allowed"] * ((((data["no_fee"] + (data["no_fee"] + (-((data["no_fee"] * (-(data["street_address"])))))))/2.0) + (-(data["num_photos"])))/2.0))).astype(float)))))) +
        0.100000*np.tanh(((((((((((data["bathrooms"] > 0.616438).astype(float)) * 2.0) - data["bathrooms"]) * ((data["bathrooms"] > data["roof"]).astype(float))) * 2.0) * 2.0) - data["site_garage"]) * 2.0) - (data["washer_in_unit"] - data["private_terrace"]))) +
        0.100000*np.tanh((((((-(((data["num_photos"] < (np.tanh(np.tanh(data["num_photos"])) - 0.700000)).astype(float)))) * 2.0) * 2.0) * 2.0) + ((data["bedrooms"] > ((data["num_photos"] > ((data["garden"] * 2.0) * 2.0)).astype(float))).astype(float)))) +
        0.100000*np.tanh(((((-(data["no_pets"])) < ((data["price"] + ((data["fitness_center"] + data["street_address"])/2.0))/2.0)).astype(float)) - ((data["price"] + (((data["no_pets"] < ((data["lowrise"] + ((data["street_address"] + data["no_pets"])/2.0))/2.0)).astype(float)) * 2.0))/2.0))) +
        0.100000*np.tanh((data["multi"] + (((((data["_photos"] > data["manager_id_mean_medium"]).astype(float)) / 2.0) - (1.162790 - ((data["deck"] < (data["manager_id_mean_medium"] + (((data["multi"] + data["deck"]) > data["manager_id_mean_medium"]).astype(float)))).astype(float)))) * 2.0))) +
        0.100000*np.tanh((data["listing_id"] * (-(np.tanh(np.tanh((data["listing_id"] + (((-(((data["exclusive"] + ((data["garden"] - data["manager_id"]) - data["fireplace"]))/2.0))) * 2.0) * 2.0)))))))) +
        0.086320*np.tanh((data["street_address"] + ((data["building_id_mean_medium"] + (data["building_id_mean_medium"] + (((((data["stainless_steel_appliances"] > (data["laundry_in_unit"] - (data["price_per_bed"] - (data["elevator"] / 2.0)))).astype(float)) * 2.0) * 2.0) * 2.0))) * data["price_per_bath"]))) +
        0.100000*np.tanh((((data["furnished"] + ((0.736842 < (((data["s_playroom"] + ((0.290323 < data["price"]).astype(float))) + ((0.290323 < data["garage"]).astype(float))) - (data["price"] / 2.0))).astype(float))) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["private_outdoor_space"] + ((((((data["bathrooms"] + data["high_ceiling"]) + data["hardwood"])/2.0) > (0.616438 - data["bedrooms"])).astype(float)) + (data["building_id"] * (((data["parking"] + data["hardwood"]) + data["no_fee"])/2.0))))) +
        0.100000*np.tanh((data["manager_id_mean_medium"] * ((data["laundry_room"] + (data["in_super"] + (((data["lowrise"] - data["num_description_words"]) + (data["no_fee"] * (data["building_id_mean_high"] + data["loft"])))/2.0))) + ((data["building_id_mean_high"] < data["loft"]).astype(float))))) +
        0.100000*np.tanh((data["wheelchair_access"] * (data["dishwasher"] * (data["patio"] + (-(((data["concierge"] + ((((np.tanh(0.736842) + ((data["wheelchair_access"] + ((data["building_id_mean_medium"] * 2.0) * 2.0))/2.0))/2.0) * 2.0) * 2.0))/2.0))))))) +
        0.100000*np.tanh(((((-(((4.052630 < (data["listing_id"] + 2.282050)).astype(float)))) * 2.0) + (((data["duplex"] > (data["price"] * (-(data["light"])))).astype(float)) - ((2.282050 < data["price"]).astype(float)))) * 2.0)) +
        0.100000*np.tanh((((((-(((data["longitude"] > ((data["fitness_center"] / 2.0) * (-(data["post_war"])))).astype(float)))) * 2.0) * 2.0) * 2.0) - (((data["laundry_in_unit"] * data["pre"]) * data["pre"]) / 2.0))) +
        0.100000*np.tanh((((((data["garage"] - data["new_construction"]) - data["simplex"]) - data["simplex"]) * data["street_address"]) - (data["simplex"] + (data["reduced_fee"] + ((data["street_address"] > ((data["building_id_mean_high"] < data["live_in_super"]).astype(float))).astype(float)))))) +
        0.100000*np.tanh(((data["latitude"] > (data["washer_"] + ((data["in_super"] + (((0.700000 + (-(np.tanh(np.tanh(np.tanh(((data["bedrooms"] > ((data["hardwood_floors"] > data["bathrooms"]).astype(float))).astype(float))))))))/2.0) / 2.0)) / 2.0))).astype(float))) +
        0.100000*np.tanh((((-(((((data["virtual_doorman"] > (data["short_term_allowed"] * ((data["short_term_allowed"] > ((-(data["latitude"])) / 2.0)).astype(float)))).astype(float)) * 2.0) * 2.0))) * 2.0) + ((data["valet"] > (-(data["latitude"]))).astype(float)))) +
        0.100000*np.tanh((((((data["duplex"] * 2.0) * 2.0) + ((data["actual_apt"] * 2.0) + data["price_per_bath"])) * 2.0) * ((data["num_photos"] - (-(data["bedrooms"]))) - ((data["reduced_fee"] + data["price_per_bath"]) * 2.0)))) +
        0.100000*np.tanh(((((data["microwave"] / 2.0) > (data["latitude"] - (-(data["longitude"])))).astype(float)) + (data["actual_apt"] - ((data["longitude"] > (data["playroom"] - (-((data["latitude"] - (data["_dishwasher_"] / 2.0)))))).astype(float))))) +
        0.100000*np.tanh((data["parking_space"] + ((data["bathrooms"] * (-(data["bathrooms"]))) + ((-(data["bathrooms"])) + ((data["work"] + ((((data["storage"] * data["bathrooms"]) < data["stainless_steel_appliances"]).astype(float)) * 2.0)) * 2.0))))) +
        0.100000*np.tanh((((((data["actual_apt"] - data["high_speed_internet"]) + data["high_ceilings"])/2.0) - (data["outdoor_areas"] - (data["high_speed_internet"] * ((data["building_id_mean_high"] < (-(((data["flex"] + (data["pets_on_approval"] * 2.0))/2.0)))).astype(float))))) * 2.0)) +
        0.100000*np.tanh(((((0.700000 > (-(((data["street_address"] + (data["_photos"] - ((((data["latitude"] * data["private_outdoor_space"]) < ((data["_photos"] > (-(data["price"]))).astype(float))).astype(float)) * 2.0))) * 2.0)))).astype(float)) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["terrace"] - ((data["latitude"] < data["war"]).astype(float))) + (data["war"] * (data["war"] * (data["building_id"] * (-((((-(((data["manager_id_mean_high"] < data["manager_id_mean_medium"]).astype(float)))) < data["manager_id_mean_high"]).astype(float))))))))) +
        0.100000*np.tanh(((((-(((np.tanh(data["num_photos"]) < (-(((data["loft"] < ((data["longitude"] + (((data["latitude"] > data["live_in_super"]).astype(float)) - (-(data["num_photos"]))))/2.0)).astype(float))))).astype(float)))) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh((((data["balcony"] * ((data["bathrooms"] * ((data["fitness_center"] + data["price_per_room"])/2.0)) - ((((data["stainless_steel_appliances"] + data["fitness_center"])/2.0) + ((data["building_id"] + data["stainless_steel_appliances"])/2.0))/2.0))) * 2.0) * 2.0)) +
        0.100000*np.tanh((data["laundry"] + (((((data["price"] - (((data["price"] - ((data["manager_id_mean_medium"] + data["price"])/2.0)) < (data["site_super"] - (1.11123585700988770))).astype(float))) < (data["sauna"] - (1.11123585700988770))).astype(float)) * 2.0) * 2.0))) +
        0.100000*np.tanh((-((((data["lounge"] + (data["latitude"] + ((((((((data["latitude"] < ((data["site_parking_lot"] * 2.0) * 2.0)).astype(float)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))) * 2.0) * 2.0)))) +
        0.100000*np.tanh(((0.134146 - ((data["latitude"] > np.tanh(0.134146)).astype(float))) - ((np.tanh(data["marble_bath"]) > (data["latitude"] + ((data["sauna"] * data["laundry_in_building"]) - (data["concierge"] / 2.0)))).astype(float)))) +
        0.100000*np.tanh((data["bedrooms"] * (((data["bedrooms"] + ((data["bedrooms"] + data["bathrooms"])/2.0)) * (data["dining_room"] - (data["price"] - ((data["bedrooms"] + data["bathrooms"])/2.0)))) - (data["private_outdoor_space"] + data["dishwasher"])))) +
        0.088260*np.tanh(((np.tanh(data["price_per_bed"]) / 2.0) - (((((data["_pets_ok_"] > (((data["price_per_bath"] - (np.tanh(data["price_per_bed"]) / 2.0)) - (data["dryer_in_unit"] / 2.0)) / 2.0)).astype(float)) * 2.0) * 2.0) * 2.0))) +
        0.100000*np.tanh((((data["price_per_bed"] * ((data["cats_allowed"] + data["private_outdoor_space"])/2.0)) + ((data["building_id_mean_high"] * (data["dogs_allowed"] + (data["hardwood"] + data["exclusive"]))) + ((data["display_address"] < (-(data["dogs_allowed"]))).astype(float))))/2.0)) +
        0.100000*np.tanh((data["outdoor_space"] * (data["exclusive"] + np.tanh(np.tanh(np.tanh(((((((-(data["manager_id"])) * 2.0) + (data["high_ceilings"] - (data["price_per_bed"] * 5.785710))) * 2.0) * 2.0) * 2.0))))))) +
        0.100000*np.tanh(((data["roof_deck"] * (((data["bedrooms"] + data["war"])/2.0) + ((data["building_id_mean_medium"] < (-(((data["bathrooms"] > (data["garden"] + data["building_id_mean_medium"])).astype(float))))).astype(float)))) + (((-(data["outdoor_areas"])) + data["bathrooms"])/2.0))) +
        0.100000*np.tanh(((-((1.0 - ((data["price_per_bath"] > data["roof"]).astype(float))))) * (data["num_description_words"] + (((data["balcony"] + (data["backyard"] + (((data["green_building"] > data["num_description_words"]).astype(float)) * 2.0))) * 2.0) * 2.0)))) +
        0.100000*np.tanh((((((data["bedrooms"] > ((data["price"] + ((2.862070 - ((data["site_parking"] - (data["building_id_mean_high"] * np.tanh(data["loft"]))) * 2.0)) / 2.0)) * 2.0)).astype(float)) * 2.0) * 2.0) * 2.0)) +
        0.100000*np.tanh(((data["building_id_mean_medium"] * data["prewar"]) + ((data["cats_allowed"] * ((data["exclusive"] + ((data["laundry_in_building"] + data["building_id_mean_medium"])/2.0))/2.0)) - ((((data["longitude"] > (-(data["indoor_pool"]))).astype(float)) > (-(data["central_a"]))).astype(float))))) +
        0.100000*np.tanh((-((data["washer_in_unit"] + ((((data["longitude"] + ((((data["_dishwasher_"] / 2.0) > (data["latitude"] - data["publicoutdoor"])).astype(float)) + ((2.282050 < data["listing_id"]).astype(float)))) * 2.0) * 2.0) * 2.0))))) +
        0.100000*np.tanh(np.tanh(((0.692308 / 2.0) - ((data["hardwood_floors"] + (data["manager_id_mean_high"] * 2.0)) * (data["building_id_mean_medium"] + (data["manager_id_mean_medium"] + (data["building_id_mean_high"] + ((2.282050 - data["manager_id_mean_high"]) / 2.0)))))))) +
        0.100000*np.tanh((data["pre"] * (((data["listing_id"] + ((data["price_per_bed"] > data["pre"]).astype(float)))/2.0) + (((data["fitness_center"] + data["live"])/2.0) + ((-((data["laundry_in_building"] - ((data["price_per_bed"] + data["virtual_doorman"])/2.0)))) / 2.0))))))
    return Outputs(p) 


def GPHighNotMedium1(data):
    p = (-1.023290 +
        0.050000*np.tanh((((data["building_id_mean_high"] - (data["manager_id_mean_high"] - (data["bathrooms"] + (data["bedrooms"] - (((data["price"] - (-1.0 + (data["concierge"] + data["manager_id_mean_high"]))) * 2.0) * 2.0))))) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["building_id_mean_high"] - (((data["price"] + ((0.144330 + data["dryer"]) + ((data["bedrooms"] < (data["marble_bath"] + data["bathrooms"])).astype(float)))) * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["hardwood_floors"] + (((((data["manager_id_mean_high"] + ((data["building_id_mean_high"] + (-(np.tanh(data["display_address"]))))/2.0)) - ((data["manager_id_mean_high"] < data["hardwood_floors"]).astype(float))) * 2.0) * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((3.571430 * (data["reduced_fee"] + (((3.0) * (data["manager_id_mean_high"] + ((data["parking"] - (data["price"] - (-((((0.439024 * 2.0) > data["manager_id_mean_high"]).astype(float)))))) * 2.0))) * 2.0)))) +
        0.050000*np.tanh(((data["manager_id_mean_medium"] + (data["no_fee"] + (data["bedrooms"] + (-(((((data["price"] - (-(((data["furnished"] < data["sublet"]).astype(float))))) * 2.0) * 2.0) - data["building_id_mean_high"])))))) * 2.0)) +
        0.050000*np.tanh(((((((data["exclusive"] + (data["dryer_in_unit"] + ((data["manager_id_mean_high"] - (data["dishwasher"] / 2.0)) - ((((data["balcony"] < data["price_per_bath"]).astype(float)) < data["price_per_bed"]).astype(float))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["building_id_mean_high"] + ((((data["manager_id_mean_high"] * 2.0) - ((data["private_outdoor_space"] > data["building_id_mean_high"]).astype(float))) * 2.0) - (((data["manager_id_mean_medium"] * 2.0) > data["building_id_mean_medium"]).astype(float)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((np.tanh(np.tanh(data["building_id_mean_high"])) - (0.315789 - ((data["site_garage"] - data["dishwasher"]) + ((((data["manager_id_mean_high"] + data["actual_apt"])/2.0) * 2.0) * 2.0)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["short_term_allowed"] + (((((data["manager_id_mean_high"] - (((data["patio"] < (data["price"] + ((data["laundry_in_unit"] < data["loft"]).astype(float)))).astype(float)) * 2.0)) - (data["price"] / 2.0)) * 2.0) * 2.0) * 2.0)) * 2.0)) +
        0.050000*np.tanh(((9.0) * (0.439024 + (data["building_id_mean_high"] - (((data["playroom"] < (-(((data["longitude"] * 2.0) * 2.0)))).astype(float)) - (data["manager_id_mean_high"] - ((data["building_id_mean_high"] < (data["longitude"] * 2.0)).astype(float)))))))) +
        0.050000*np.tanh(((((((((((data["no_fee"] * 0.538462) + (data["exclusive"] + (-(0.315789)))) - (0.315789 + data["price"])) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["building_id_mean_high"] + ((-(((data["price"] / 2.0) - (data["manager_id_mean_high"] - (((data["terrace"] < ((data["decorative_fireplace"] + (data["price"] / 2.0))/2.0)).astype(float)) * 2.0))))) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["no_fee"] + (data["bedrooms"] + ((data["manager_id_mean_high"] - (((1.545450 > (data["granite_kitchen"] - data["price"])).astype(float)) * 2.0)) - (data["price"] + data["price"]))))) +
        0.050000*np.tanh(((((((((data["manager_id_mean_high"] - ((data["no_fee"] < (data["price"] / 2.0)).astype(float))) - (data["manager_id_mean_high"] / 2.0)) - ((data["bedrooms"] < data["pet_friendly"]).astype(float))) - data["price"]) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((data["manager_id_mean_high"] - (((data["furnished"] > data["manager_id_mean_high"]).astype(float)) - (((data["short_term_allowed"] + (data["furnished"] - (data["price"] + ((data["price_per_room"] > data["short_term_allowed"]).astype(float))))) * 2.0) * 2.0))) * 2.0) * 2.0)) +
        0.050000*np.tanh((((data["building_id_mean_high"] + ((((data["manager_id_mean_high"] - (((data["furnished"] < (-(data["building_id_mean_high"]))).astype(float)) * 2.0)) * 2.0) * 2.0) - (data["building_id"] + ((data["private_outdoor_space"] + data["building_id_mean_high"])/2.0)))) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["furnished"] + (data["no_fee"] + (data["bedrooms"] + (data["manager_id_mean_high"] - (((data["price"] + ((data["bedrooms"] < ((data["price"] + ((data["price"] < 3.172410).astype(float)))/2.0)).astype(float))) * 2.0) * 2.0)))))) +
        0.050000*np.tanh((((((((data["no_fee"] + (np.tanh(np.tanh(data["bedrooms"])) - (data["price"] + (data["price"] + 2.071430)))) * 2.0) - data["price"]) * 2.0) - data["granite_kitchen"]) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((((data["central_a"] - (((((data["price"] + (data["price_per_bed"] - (data["manager_id_mean_high"] / 2.0)))/2.0) * 2.0) + (data["price_per_bed"] * data["price_per_bed"]))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["price"] - (((((((((data["price"] + ((data["private_terrace"] < data["longitude"]).astype(float))) - ((data["site_super"] > data["longitude"]).astype(float))) * 2.0) * 2.0) - data["price"]) * 2.0) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((data["furnished"] + ((data["common_outdoor_space"] - ((data["no_fee"] < data["furnished"]).astype(float))) - (data["display_address"] - (-((((data["price"] + ((data["price"] < data["stainless_steel_appliances"]).astype(float))) * 2.0) * 2.0))))))) +
        0.050000*np.tanh((((((((-(np.tanh((data["price_per_bed"] + data["display_address"])))) * 2.0) + (data["manager_id_mean_medium"] - ((data["manager_id_mean_high"] + data["price"])/2.0))) + (data["manager_id_mean_high"] - data["live_in_super"])) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((np.tanh((data["cats_allowed"] - data["subway"])) - ((((data["price"] + ((2.875000 > (data["dining_room"] + data["building_id_mean_high"])).astype(float)))/2.0) * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((np.tanh((data["furnished"] - (np.tanh(((data["in_super"] > data["multi"]).astype(float))) - ((data["longitude"] < data["backyard"]).astype(float))))) * 2.0) * 2.0) + data["manager_id_mean_high"]) * 2.0) * 2.0) + data["actual_apt"])) +
        0.050000*np.tanh((data["hardwood_floors"] + ((data["wheelchair_access"] + (((data["exclusive"] + ((((data["terrace"] - data["roof_deck"]) - (data["manager_id"] * 2.0)) + (data["manager_id_mean_high"] - data["hardwood"]))/2.0)) * 2.0) * 2.0)) * 2.0))) +
        0.050000*np.tanh(((((((-((data["granite_kitchen"] + (data["doorman"] - (1.809520 * data["building_id_mean_high"]))))) - (-(((data["common_outdoor_space"] + ((data["price_per_room"] < data["luxury_building"]).astype(float)))/2.0)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((-(((data["price"] + (data["s_kitchen_"] + ((((data["bedrooms"] > -1.0).astype(float)) > ((data["longitude"] < data["s_kitchen_"]).astype(float))).astype(float)))) - data["bedrooms"]))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((data["bedrooms"] + (data["furnished"] - (data["price"] + (np.tanh(((data["full_service_garage"] < (data["longitude"] * 2.0)).astype(float))) * 2.0)))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((np.tanh((-((1.543480 + data["price"])))) * 2.0) * 2.0) - data["display_address"]) + (data["balcony"] - ((data["price"] - data["short_term_allowed"]) - (data["laundry_in_unit"] * 2.0))))) +
        0.050000*np.tanh((((np.tanh(data["price"]) - (-((data["no_fee"] - (((((data["price"] + data["hardwood"]) + ((data["bedrooms"] < ((data["loft"] < data["price"]).astype(float))).astype(float)))/2.0) * 2.0) * 2.0))))) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((data["in_superintendent"] / 2.0) < (-(data["latitude"]))).astype(float)) * 2.0) - (data["price"] + (((-(data["site_super"])) < data["latitude"]).astype(float)))) - ((((data["longitude"] > data["latitude"]).astype(float)) * 2.0) * 2.0))) +
        0.050000*np.tanh((((((((((data["manager_id_mean_high"] + data["manager_id_mean_medium"]) / 2.0) / 2.0) - ((0.708861 > ((data["actual_apt"] > (((data["longitude"] * 2.0) * 2.0) * 2.0)).astype(float))).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["furnished"] - (data["doorman"] + ((((data["price"] + ((data["residents_lounge"] < ((data["longitude"] * 2.0) * 2.0)).astype(float))) * 2.0) + (((data["s_kitchen_"] < data["longitude"]).astype(float)) - data["bathrooms"])) * 2.0)))) +
        0.050000*np.tanh(((((((data["no_fee"] - (data["hardwood"] + (data["dishwasher"] + ((1.543480 > (-(data["price"]))).astype(float))))) + ((data["pre"] < (data["num_photos"] * 2.0)).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["laundry_in_building"] + ((((data["common_outdoor_space"] + (-((1.543480 + ((data["price"] - (data["short_term_allowed"] - (((data["laundry_in_building"] < data["street_address"]).astype(float)) / 2.0))) * 2.0))))) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((data["manager_id_mean_medium"] - (((((data["price_per_room"] * ((-1.0 < data["manager_id_mean_medium"]).astype(float))) + data["display_address"]) * 2.0) * 2.0) + (data["hardwood"] + (data["price"] + (data["street_address"] + data["price"])))))) +
        0.050000*np.tanh(((data["elevator"] * (((data["num_photos"] * (data["num_photos"] * 2.0)) + (((data["num_photos"] * data["bathrooms"]) + (data["no_fee"] * 2.0)) * 2.0)) + (-((data["hardwood"] * 2.0))))) * 2.0)) +
        0.050000*np.tanh(((((data["furnished"] + data["new_construction"]) + (-(data["price"]))) - (data["street_address"] - data["no_fee"])) - (((((1.545450 + (data["hardwood"] + data["price"]))/2.0) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((data["furnished"] - (data["price"] - ((data["dryer_in_unit"] - (0.538462 - ((((data["longitude"] < ((data["concierge"] + (data["concierge"] * np.tanh(data["price_per_room"])))/2.0)).astype(float)) * 2.0) * 2.0))) * 2.0)))) +
        0.050000*np.tanh((3.571430 * (3.571430 * ((data["manager_id_mean_medium"] - (data["swimming_pool"] - ((-((((-(((data["latitude"] < data["luxury_building"]).astype(float)))) < data["longitude"]).astype(float)))) * 2.0))) - data["latitude"])))) +
        0.050000*np.tanh((data["elevator"] * (((((((np.tanh(data["laundry_in_building"]) + (data["display_address"] + (data["price_per_room"] * (data["high_speed_internet"] - np.tanh(data["laundry_in_building"]))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((data["bedrooms"] - (data["price"] + ((-(data["reduced_fee"])) + (1.0 - (((data["renovated"] > (-((data["bedrooms"] * (data["furnished"] * data["fitness_center"]))))).astype(float)) * 2.0)))))) +
        0.050000*np.tanh(((((data["reduced_fee"] - ((data["air_conditioning"] - (data["building_id_mean_medium"] * ((data["dogs_allowed"] + (data["num_description_words"] - data["simplex"])) * 2.0))) - ((data["price_per_bath"] < data["building_id_mean_medium"]).astype(float)))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["bedrooms"] - ((data["price"] + (((data["no_pets"] > (data["latitude"] - ((((data["site_super"] > (data["air_conditioning"] - data["longitude"])).astype(float)) > (data["actual_apt"] - data["swimming_pool"])).astype(float)))).astype(float)) * 2.0)) * 2.0))) +
        0.050000*np.tanh((data["laundry_in_building"] - (data["price"] - (data["manager_id_mean_medium"] * (data["manager_id_mean_medium"] * (data["manager_id_mean_medium"] * ((data["building_id_mean_high"] + (data["price"] - ((data["street_address"] * 2.0) * 2.0)))/2.0))))))) +
        0.050000*np.tanh(((((((((data["latitude"] > data["longitude"]).astype(float)) - (data["pre"] + (data["price"] + (((data["wheelchair_access"] > (data["price"] + data["pre"])).astype(float)) * 2.0)))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["laundry_in_unit"] - (data["building_id"] + ((((((data["price_per_room"] > data["building_id"]).astype(float)) - ((data["building_id"] < (data["_photos"] - ((data["price_per_room"] > data["parking_space"]).astype(float)))).astype(float))) * 2.0) * 2.0) * 2.0)))) +
        0.050000*np.tanh((((1.809520 * 2.0) * 2.0) * (((data["price_per_bed"] * (((data["flex"] / 2.0) < (data["latitude"] * (data["latitude"] * (-(1.809520))))).astype(float))) < (data["walk_in_closet"] / 2.0)).astype(float)))) +
        0.050000*np.tanh((((((((-(data["price"])) + (-(((-(np.tanh((data["bedrooms"] - (-(data["no_fee"])))))) + ((data["manager_id_mean_high"] > data["common_outdoor_space"]).astype(float)))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((-(((((data["no_fee"] * 2.0) * ((data["_photos"] + (data["dogs_allowed"] + (data["exclusive"] + ((data["wheelchair_access"] + data["laundry_in_building"])/2.0))))/2.0)) + (data["doorman"] * (data["dishwasher"] * 2.0))) * 2.0)))))
    return Outputs(p)


def GPHighNotMedium2(data):
    p = (-1.023290 +
        0.050000*np.tanh((((((data["building_id_mean_high"] + data["laundry_in_unit"]) + (((data["manager_id_mean_high"] - ((((data["manager_id_mean_high"] < 1.0).astype(float)) + data["price"]) * 2.0)) * 2.0) * 2.0)) + data["price"]) + data["short_term_allowed"]) * 2.0)) +
        0.050000*np.tanh((((((((data["manager_id_mean_high"] - ((data["price"] - (data["walk"] - ((((1.789470 + data["microwave"])/2.0) > data["manager_id_mean_high"]).astype(float)))) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["building_id_mean_medium"] + (-(((np.tanh(((((data["price"] - data["manager_id_mean_high"]) + (-(np.tanh((data["building_id_mean_high"] - 4.250000)))))/2.0) * 2.0)) * 2.0) * 2.0)))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((data["walk"] + (data["bedrooms"] + (data["building_id_mean_high"] + (((data["manager_id_mean_high"] - ((data["price"] + 1.246580) + 1.246580)) - data["price"]) * 2.0)))) * 2.0) + 1.246580)) +
        0.050000*np.tanh(((((4.250000 * ((data["building_id_mean_high"] - 0.563830) - (-(((data["manager_id_mean_high"] + ((data["elevator"] < (data["exclusive"] - data["war"])).astype(float)))/2.0))))) * 2.0) * 4.250000) * 4.250000)) +
        0.050000*np.tanh(((((data["price"] - (-((((data["manager_id_mean_high"] + (-(((data["price"] + ((data["bedrooms"] < ((data["manager_id_mean_high"] < data["price"]).astype(float))).astype(float))) * 2.0)))) * 2.0) * 2.0)))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((2.227270 * 2.0) + (data["price"] + ((data["furnished"] - ((((data["price"] + ((1.916670 + ((data["exclusive"] < (-(data["no_fee"]))).astype(float)))/2.0)) * 2.0) * 2.0) * 2.0)) * 2.0)))) +
        0.050000*np.tanh((((((data["bedrooms"] + (data["no_fee"] - (data["manager_id_mean_high"] - ((data["manager_id_mean_high"] + ((data["short_term_allowed"] - ((0.816667 + data["price"])/2.0)) * 2.0)) * 2.0)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["building_id_mean_medium"] + ((data["exclusive"] - data["hardwood_floors"]) - data["roof_deck"]))/2.0) * 2.0) + (((data["manager_id_mean_high"] - ((data["pets_on_approval"] > (data["manager_id_mean_high"] - data["elevator"])).astype(float))) * 2.0) * 2.0)) * 2.0)) +
        0.050000*np.tanh((-((((0.836735 - (data["building_id_mean_high"] + (data["manager_id_mean_high"] - ((data["price"] + ((data["price_per_room"] > (-(((data["latitude"] > (-(data["price_per_bed"]))).astype(float))))).astype(float))) * 2.0)))) * 2.0) * 2.0)))) +
        0.050000*np.tanh(((((((data["manager_id_mean_high"] - np.tanh(((data["private_backyard"] < data["longitude"]).astype(float)))) + (np.tanh(((-1.0 + (-(data["price"]))) * 2.0)) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["building_id_mean_high"] + (data["manager_id_mean_high"] + (((data["price"] + ((((data["furnished"] * 2.0) - (((data["common_outdoor_space"] < (-(data["bedrooms"]))).astype(float)) + data["price"])) * 2.0) * 2.0))/2.0) * 2.0))) * 2.0)) +
        0.050000*np.tanh((((data["exclusive"] + ((data["manager_id_mean_high"] + (((((-(((data["price"] > (((data["bedrooms"] + data["s_playroom"])/2.0) * 2.0)).astype(float)))) * 2.0) * 2.0) + (-(data["price"])))/2.0)) * 2.0)) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((((data["building_id_mean_high"] + ((-((data["price"] - ((data["level"] + ((data["level"] * 2.0) * 2.0)) * 2.0)))) * 2.0)) * 2.0) * 2.0) + data["new_construction"]) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["price"] + ((((np.tanh((-((0.836735 + (-(data["bedrooms"])))))) - data["price"]) - data["marble_bath"]) * 2.0) * 2.0)) + data["no_fee"]) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["short_term_allowed"] + (data["parking"] + (((data["manager_id_mean_medium"] - (data["price"] + ((1.361700 > data["building_id_mean_medium"]).astype(float)))) + (data["manager_id_mean_high"] * 2.0)) * 2.0))) - data["price"]) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["manager_id_mean_high"] - (((data["price"] - ((((data["longitude"] < data["video_intercom"]).astype(float)) - ((data["shares_ok"] < data["price_per_bed"]).astype(float))) - ((data["price"] < 0.0).astype(float)))) * 2.0) * 2.0)) * 2.0)) +
        0.050000*np.tanh(((13.80861186981201172) * (data["exclusive"] + np.tanh((data["bedrooms"] + (((((data["furnished"] + (data["no_fee"] - data["hardwood"]))/2.0) + np.tanh((-1.0 - data["price"]))) * 2.0) * 2.0)))))) +
        0.050000*np.tanh(((((((((data["bedrooms"] - ((1.246580 + data["subway"])/2.0)) + data["manager_id_mean_high"])/2.0) + (((data["building_id_mean_high"] + data["exclusive"])/2.0) + (data["short_term_allowed"] - data["display_address"]))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["common_outdoor_space"] - (data["hardwood"] - ((data["loft"] + (((data["manager_id_mean_high"] + ((data["manager_id_mean_high"] * data["doorman"]) - data["loft"])) * 2.400000) - data["price_per_bed"])) * 4.250000)))) +
        0.050000*np.tanh((-((((data["street_address"] - (data["no_fee"] - ((((data["price"] - (data["24"] - ((1.361700 > (data["manager_id_mean_high"] / 2.0)).astype(float)))) * 2.0) * 2.0) - -1.0))) * 2.0) * 2.0)))) +
        0.050000*np.tanh(((12.74075984954833984) * ((12.74076366424560547) * ((data["renovated"] + ((data["manager_id_mean_high"] + ((((data["common_outdoor_space"] + (((data["latitude"] > (-(data["building_id_mean_high"]))).astype(float)) / 2.0)) > data["building_id"]).astype(float)) / 2.0)) * 2.0))/2.0)))) +
        0.050000*np.tanh((((data["renovated"] + (data["furnished"] + (((((-(((data["courtyard"] > ((data["building_id_mean_high"] + data["reduced_fee"]) * 2.0)).astype(float)))) + (-(data["renovated"]))) * 2.0) * 2.0) * 2.0))) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((((data["terrace"] + ((data["private_balcony"] + data["price"])/2.0))/2.0) - (data["price"] + (data["longitude"] + ((data["playroom"] < (data["longitude"] + data["longitude"])).astype(float))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((-((data["price"] - (((data["furnished"] + (-((data["price"] - (data["laundry_in_unit"] - ((data["no_fee"] < data["price_per_bath"]).astype(float)))))))/2.0) - 1.246580)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["manager_id_mean_medium"] - (data["subway"] + (((data["lowrise"] + (data["live_in_super"] + ((data["dishwasher"] - (data["manager_id_mean_medium"] - data["display_address"])) - (data["building_id_mean_high"] - data["building_id"])))) * 2.0) * 2.0)))) +
        0.050000*np.tanh((-(((((data["price"] + (1.916670 * np.tanh((data["price_per_room"] + (data["latitude"] - (-((((data["dryer"] / 2.0) > (-(data["latitude"]))).astype(float))))))))) * 2.0) * 2.0) * 2.0)))) +
        0.050000*np.tanh((data["no_fee"] - (((((data["price"] + (data["price"] + ((0.563830 + ((data["no_fee"] < data["marble_bath"]).astype(float))) + ((data["high_speed_internet"] < data["price_per_room"]).astype(float)))))/2.0) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh(((((data["actual_apt"] + (((data["health_club"] + ((data["washer_"] + (data["furnished"] + ((data["hi_rise"] > data["longitude"]).astype(float)))) * 2.0)) + data["dryer_in_unit"]) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["common_outdoor_space"] - (np.tanh((((data["longitude"] > (data["_photos"] / 2.0)).astype(float)) + ((data["doorman"] + (data["private_outdoor_space"] + ((data["price"] + data["doorman"])/2.0)))/2.0))) * 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["no_fee"] + (-(((data["price_per_bath"] - (data["price_per_room"] / 2.0)) - (np.tanh(((data["_photos"] > (data["price_per_room"] - data["short_term_allowed"])).astype(float))) * 2.0))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["price"] + ((data["furnished"] + (data["cats_allowed"] + ((((data["reduced_fee"] + (data["bedrooms"] - (data["price"] * (data["no_fee"] - data["price"]))))/2.0) * 2.0) * 2.0))) * 2.0))) +
        0.050000*np.tanh((-(((((((((((data["price"] + ((data["price"] < np.tanh(data["building_id_mean_high"])).astype(float)))/2.0) * 2.0) * 2.0) + np.tanh(data["manager_id"])) * 2.0) + np.tanh(data["manager_id"])) * 2.0) * 2.0) * 2.0)))) +
        0.050000*np.tanh(((((((0.836735 > (data["prewar"] - (-(data["price_per_bed"])))).astype(float)) - (data["price"] - (-((((data["longitude"] > ((data["actual_apt"] / 2.0) / 2.0)).astype(float)) * 2.0))))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["balcony"] + (data["balcony"] + (data["laundry_in_building"] - (((data["price"] + ((data["hardwood"] > data["bedrooms"]).astype(float))) + data["building_id_mean_medium"]) + (data["price"] + ((data["price"] > data["bedrooms"]).astype(float)))))))) +
        0.050000*np.tanh(((((((data["manager_id_mean_medium"] - ((data["roof"] + ((data["longitude"] > ((data["air_conditioning"] / 2.0) + (data["roof_deck"] * data["ft_doorman"]))).astype(float))) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["furnished"] - (((data["fitness_center"] - ((data["manager_id_mean_high"] + (((-(((data["price_per_room"] > (data["duplex"] * 2.0)).astype(float)))) * 2.0) * 2.0))/2.0)) * 2.0) * 2.0))) +
        0.050000*np.tanh(((((((data["bike_room"] > (data["longitude"] * 2.0)).astype(float)) * 2.0) * 2.0) * 2.0) - (data["price"] + ((data["price"] + ((((((data["midrise"] < data["building_id_mean_high"]).astype(float)) > data["longitude"]).astype(float)) * 2.0) * 2.0))/2.0)))) +
        0.050000*np.tanh(((((((-1.0 * ((data["stainless_steel_appliances"] > (data["latitude"] * 2.0)).astype(float))) * 2.0) - np.tanh((data["display_address"] - ((((data["_photos"] < data["building_id"]).astype(float)) < data["manager_id_mean_medium"]).astype(float))))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((data["furnished"] - (((1.916670 / 2.0) + (data["price"] + ((data["valet"] + (-(data["laundry_in_unit"])))/2.0)))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["manager_id_mean_high"] * ((data["bedrooms"] + 1.361700)/2.0)) - (data["street_address"] + ((((data["no_fee"] < ((data["cats_allowed"] + data["manager_id_mean_high"]) * ((data["swimming_pool"] * 2.0) * 2.0))).astype(float)) * 2.0) * 2.0)))) +
        0.050000*np.tanh((((((-(((data["latitude"] < data["longitude"]).astype(float)))) + (np.tanh((((data["num_photos"] + (data["num_photos"] - (data["pre"] + data["hardwood"]))) * 2.0) * 2.0)) / 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["subway"] * data["swimming_pool"]) + ((data["common_outdoor_space"] - (data["price"] + (data["price"] + ((data["latitude"] < data["subway"]).astype(float))))) - (data["price_per_room"] + ((data["latitude"] < data["work"]).astype(float)))))) +
        0.050000*np.tanh((((-((data["price"] - (data["bedrooms"] + (data["furnished"] - (((data["longitude"] > data["pet_friendly"]).astype(float)) + ((data["longitude"] > ((data["_photos"] > data["bedrooms"]).astype(float))).astype(float)))))))) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["furnished"] - ((((1.0 / 2.0) - (((-(data["latitude"])) > ((data["price_per_bed"] + ((data["latitude"] < data["residents_garden"]).astype(float))) + ((data["latitude"] < data["marble_bath"]).astype(float)))).astype(float))) * 2.0) * 2.0))) +
        0.050000*np.tanh((((((data["fitness_center"] * ((data["parking_space"] + (data["price_per_bath"] + data["price_per_bath"])) * 2.0)) + (((data["latitude"] + (data["parking_space"] / 2.0)) < (-(data["longitude"]))).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["cats_allowed"] * ((((data["price_per_room"] < ((-(((data["dogs_allowed"] < data["garden"]).astype(float)))) / 2.0)).astype(float)) * (data["doorman"] - data["price_per_bed"])) + data["price_per_bath"])) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["common_outdoor_space"] + (-((data["price"] + (((np.tanh((data["manager_id_mean_high"] - data["wheelchair_access"])) * 2.0) > (data["laundry_in_building"] * data["elevator"])).astype(float)))))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["bedrooms"] + ((data["no_fee"] + (((data["storage"] > np.tanh((data["bike_room"] * (-(data["price"]))))).astype(float)) * 2.0)) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((-(np.tanh((data["swimming_pool"] * (((data["latitude"] > data["site_super"]).astype(float)) + data["longitude"]))))) > ((((((data["latitude"] + data["subway"])/2.0) > (-(data["longitude"]))).astype(float)) > (-(data["laundry_room"]))).astype(float))).astype(float))))
    return Outputs(p)


def GPHighNotMedium3(data):
    p = (-1.023290 +
        0.050000*np.tanh(((((-(data["price"])) + (((data["reduced_fee"] + (np.tanh((data["building_id_mean_high"] - (-((((-(data["price"])) - 1.174600) * 2.0))))) * 2.0)) * 2.0) * 2.0)) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["building_id_mean_high"] + ((data["bedrooms"] - (data["price"] - (((data["manager_id_mean_high"] - (1.174600 + (data["price"] - (data["bedrooms"] * (data["manager_id_mean_high"] / 2.0))))) * 2.0) * 2.0))) * 2.0))) +
        0.050000*np.tanh(((((data["building_id_mean_high"] - data["simplex"]) + (((-((0.843137 - (data["manager_id_mean_high"] - ((data["price"] + ((data["hardwood"] < data["price_per_room"]).astype(float))) * 2.0))))) * 2.0) * 2.0)) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["price"] + ((((data["manager_id_mean_high"] - (data["price"] + (((((0.843137 > data["manager_id_mean_high"]).astype(float)) * 2.0) + data["manager_id_mean_high"])/2.0))) * 2.0) * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["building_id_mean_high"] - (((data["price"] - (data["short_term_allowed"] - ((data["price"] + (0.692308 - (data["manager_id_mean_high"] - ((data["furnished"] < (-(data["bedrooms"]))).astype(float))))) * 2.0))) * 2.0) * 2.0))) +
        0.050000*np.tanh(((((((((data["short_term_allowed"] + (data["manager_id_mean_high"] - np.tanh(((2.282050 * (data["price"] + 0.843137)) + 0.736842)))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["building_id_mean_high"] + ((data["manager_id_mean_high"] - ((data["price"] - -1.0) * 2.0)) * 2.0)) + ((data["laundry_in_unit"] + data["bedrooms"]) + data["short_term_allowed"])) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((9.52254486083984375) * ((data["new_construction"] + (((((data["manager_id_mean_high"] * 2.0) - data["dishwasher"]) * 2.0) - ((data["multi"] < data["doorman"]).astype(float))) * 2.0)) - data["subway"])) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((((((data["washer"] - (1.0 - (((data["building_id_mean_high"] * 2.0) * 2.0) + data["building_id_mean_medium"]))) * 2.0) - data["doorman"]) * 2.0) * 2.0) - data["doorman"]) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["building_id_mean_high"] - ((data["price"] - data["bedrooms"]) - (((data["manager_id_mean_high"] - (data["price"] - data["no_fee"])) - ((((-(data["exclusive"])) * 2.0) * 2.0) * 2.0)) * 2.0))) * 2.0)) +
        0.050000*np.tanh((((((((data["manager_id_mean_high"] - (0.600000 + data["price"])) - (1.174600 + ((1.174600 + data["price"]) + (-(data["building_id_mean_high"]))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((((-1.0 - (data["price"] - ((((data["manager_id_mean_high"] + data["laundry_in_unit"])/2.0) + (data["furnished"] * 2.0))/2.0))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((data["no_fee"] + (((data["exclusive"] - (data["price"] + ((data["price_per_room"] > data["hardwood"]).astype(float)))) * 2.0) * 2.0)) - data["hardwood"]) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((data["building_id_mean_medium"] + (data["no_fee"] + ((((np.tanh(np.tanh(np.tanh((((data["manager_id_mean_high"] * 2.0) * 2.0) * 2.0)))) * 2.0) + (-(data["price"]))) * 2.0) * 2.0))) * 2.0)) +
        0.050000*np.tanh(((((((data["short_term_allowed"] - data["price_per_bed"]) + ((data["manager_id_mean_high"] + (((-(((data["price_per_room"] > data["site_garage"]).astype(float)))) * 2.0) * 2.0)) - data["price"])) - data["price"]) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["furnished"] - (data["price_per_room"] - (data["building_id_mean_high"] + ((data["manager_id_mean_high"] + (((-(data["price"])) + ((-((((data["longitude"] * 2.0) > data["childrens_playroom"]).astype(float)))) * 2.0)) * 2.0)) * 2.0))))) +
        0.050000*np.tanh(((((((((data["level"] - (-(np.tanh((data["building_id_mean_high"] * 2.0))))) * 2.0) + data["manager_id_mean_high"]) + data["manager_id_mean_high"]) + (-(data["display_address"]))) + (-(data["building_id"]))) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["building_id_mean_high"] + (((((data["manager_id_mean_high"] + ((-(((data["high_ceiling"] > (np.tanh(data["bedrooms"]) + (-(data["display_address"])))).astype(float)))) * 2.0)) + (-(data["hardwood"]))) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((((((data["reduced_fee"] - (data["price"] + 0.700000)) - (((0.843137 > data["price_per_room"]).astype(float)) - (-((data["price"] - (data["bedrooms"] + data["no_fee"])))))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((data["furnished"] + (((data["common_outdoor_space"] - ((data["price"] + ((data["no_fee"] < data["marble_bath"]).astype(float)))/2.0)) * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((data["manager_id_mean_high"] + (((((data["common_outdoor_space"] - (((data["manager_id_mean_high"] < data["loft"]).astype(float)) - ((data["latitude"] < (data["site_laundry"] * data["dryer"])).astype(float)))) * 2.0) * 2.0) * 2.0) * 2.0)) * 2.0) * 2.0)) +
        0.050000*np.tanh(((8.0) + ((14.05183601379394531) * ((14.05183601379394531) * ((((data["manager_id_mean_high"] > (data["walk_in_closet"] * 2.0)).astype(float)) - (data["price_per_bed"] + ((data["doorman"] > (data["price_per_bath"] - data["doorman"])).astype(float)))) * 2.0))))) +
        0.050000*np.tanh((data["manager_id_mean_high"] + (data["multi"] + (data["exclusive"] - (data["hardwood_floors"] - ((data["hardwood_floors"] * (data["no_fee"] * (data["hardwood_floors"] + ((data["manager_id_mean_medium"] * 2.0) * 2.0)))) + data["multi"])))))) +
        0.050000*np.tanh(((((((data["furnished"] + data["building_id_mean_high"]) - (((data["display_address"] - (((data["bedrooms"] / 2.0) - data["subway"]) + data["renovated"])) * 2.0) - data["manager_id_mean_medium"])) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((data["furnished"] + (data["actual_apt"] + ((-(data["price"])) - ((data["longitude"] > (data["site_parking"] / 2.0)).astype(float))))) * 2.0) - ((data["price_per_room"] > data["roof_deck"]).astype(float))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["_dryer"] - (5.785710 * (5.785710 * (data["publicoutdoor"] - (((data["longitude"] < data["s_kitchen_"]).astype(float)) - ((data["publicoutdoor"] < ((data["price"] + 1.162790)/2.0)).astype(float)))))))) +
        0.050000*np.tanh((data["building_id_mean_high"] - ((((data["manager_id"] - (data["terrace"] - ((data["price"] + (data["granite_kitchen"] + ((data["wheelchair_access"] < (data["building_id_mean_high"] - data["price"])).astype(float)))) * 2.0))) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh(((((data["price_per_room"] + ((4.333330 * (((-(((((data["_photos"] + data["furnished"])/2.0) < data["price_per_room"]).astype(float)))) * 2.0) + (-(data["price"])))) / 2.0)) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((-((data["price_per_bed"] - (data["balcony"] + (((-(data["price"])) + ((data["balcony"] - data["street_address"]) + ((data["building_id_mean_high"] - data["hardwood"]) - data["price_per_room"]))) * 2.0))))) * 2.0)) +
        0.050000*np.tanh(((((((((((data["longitude"] < (data["actual_apt"] / 2.0)).astype(float)) - (data["longitude"] + ((0.117647 + ((data["price"] + 1.174600)/2.0))/2.0))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["exclusive"] + (data["bedrooms"] + (data["cats_allowed"] - (-((((data["furnished"] + (data["bedrooms"] + (data["exclusive"] - (-((data["no_fee"] - data["price"])))))) * 2.0) * 2.0))))))) +
        0.050000*np.tanh((((((((data["laundry_in_building"] + (np.tanh(((data["actual_apt"] - (data["roofdeck"] + (data["price"] + 1.162790))) * 2.0)) * 2.0)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((2.862070 * (2.862070 * (data["manager_id_mean_medium"] + (data["furnished"] - (data["display_address"] + ((data["high_ceilings"] + ((data["laundry_in_building"] < data["manager_id_mean_medium"]).astype(float)))/2.0)))))) - (data["roof_deck"] * 2.0))) +
        0.050000*np.tanh((-(((((data["price"] - ((-1.0 + (data["bedrooms"] - (data["price"] - (data["cats_allowed"] * data["building_id_mean_medium"])))) * 2.0)) - data["no_fee"]) - data["no_fee"]) * 2.0)))) +
        0.050000*np.tanh((data["bedrooms"] - ((data["prewar"] - data["manager_id_mean_high"]) + (-((5.785710 * (5.785710 * (data["bedrooms"] - (data["price"] + ((data["pre"] > data["manager_id_mean_high"]).astype(float))))))))))) +
        0.050000*np.tanh(((((((((((data["longitude"] < data["_pets_ok_"]).astype(float)) - ((data["num_photos"] < ((-1.0 - data["multi"]) / 2.0)).astype(float))) * 2.0) - data["num_photos"]) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((data["furnished"] - (data["price"] + (data["price_per_bath"] + ((data["actual_apt"] < (data["wheelchair_access"] * (((data["no_fee"] > data["high_speed_internet"]).astype(float)) - (data["price_per_bath"] * 2.0)))).astype(float))))) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["manager_id_mean_high"] - (data["building_id_mean_medium"] - (-((data["manager_id"] + (data["price"] + (((data["price_per_bath"] * (((data["manager_id_mean_high"] > data["fitness_center"]).astype(float)) * 2.0)) * 2.0) * 2.0)))))))) +
        0.050000*np.tanh((data["common_outdoor_space"] - (data["price"] - ((((data["_photos"] - (0.692308 - ((data["private_outdoor_space"] < ((-(0.117647)) + (-(data["longitude"])))).astype(float)))) * 2.0) - data["fitness_center"]) * 2.0)))) +
        0.050000*np.tanh((((-(((((data["swimming_pool"] + (data["price"] + (data["price_per_room"] + ((((-(((data["latitude"] < data["outdoor_areas"]).astype(float)))) < data["longitude"]).astype(float)) * 2.0)))) * 2.0) * 2.0) * 2.0))) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((data["laundry_in_building"] - (data["elevator"] - (-((((data["manager_id_mean_medium"] < data["bathrooms"]).astype(float)) * (((data["multi"] < data["green_building"]).astype(float)) * (10.51700878143310547))))))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((((((data["dryer_in_unit"] - ((data["price"] + ((8.0) * data["latitude"]))/2.0)) - ((data["latitude"] < data["longitude"]).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((((((((-(((data["building"] > (data["latitude"] - ((data["publicoutdoor"] > (data["num_photos"] / 2.0)).astype(float)))).astype(float)))) + (-(data["latitude"]))) * 2.0) * 2.0) + (-(data["pre"]))) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["dogs_allowed"] * ((((((((data["building_id_mean_medium"] - (data["bedrooms"] + data["bedrooms"])) - (data["reduced_fee"] + data["display_address"])) * 2.0) - data["dogs_allowed"]) - data["bedrooms"]) * 2.0) * 2.0) * 2.0))) +
        0.050000*np.tanh((data["no_fee"] - ((data["price"] + (((0.290323 - (data["bedrooms"] - (data["price"] + (((data["street_address"] + (data["hardwood"] - data["no_fee"]))/2.0) * 2.0)))) * 2.0) * 2.0)) * 2.0))) +
        0.050000*np.tanh(((((((((-((((data["latitude"] > (-(((((data["roofdeck"] > data["latitude"]).astype(float)) + data["walk_in_closet"])/2.0)))).astype(float)) + data["longitude"]))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh((data["short_term_allowed"] - (data["manager_id"] + ((-(data["fitness_center"])) + (data["price"] + (-(((((-((data["price"] + ((data["marble_bath"] > data["high_speed_internet"]).astype(float))))) * 2.0) * 2.0) * 2.0)))))))) +
        0.050000*np.tanh((data["num_description_words"] + (np.tanh(((data["wheelchair_access"] - (data["roof_deck"] + ((data["war"] + (((data["swimming_pool"] + ((data["wifi_access"] > (data["latitude"] * 2.0)).astype(float))) * 2.0) * 2.0))/2.0))) * 2.0)) * 2.0))) +
        0.050000*np.tanh(((((((data["laundry_in_building"] * data["elevator"]) - (((data["elevator"] > data["price_per_bed"]).astype(float)) + (((data["price_per_bed"] > data["24"]).astype(float)) + (data["light"] * data["elevator"])))) * 2.0) * 2.0) * 2.0) * 2.0)) +
        0.050000*np.tanh(((((((data["furnished"] - (data["building_id"] * (data["manager_id"] * data["manager_id"]))) + (data["garden"] + (data["garden"] + (data["display_address"] * data["manager_id"])))) * 2.0) * 2.0) * 2.0) * 2.0)))
    return Outputs(p)


def GPNotLo(data):
    return (GPNotLo1(data)+GPNotLo2(data)+GPNotLo3(data))/3.


def GPHighNotMedium(data):
    return (GPHighNotMedium1(data)+GPHighNotMedium2(data)+GPHighNotMedium2(data))/3.


if __name__ == "__main__":
    myseed=321
    random.seed(myseed)
    np.random.seed(myseed)

    X_train = pd.read_json("../input/train.json")
    X_test = pd.read_json("../input/test.json")

    interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
    X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])
    X_test['interest_level'] = -1
    X_train.price = np.log(X_train.price)
    X_test.price = np.log(X_test.price)
    feature_transform = CountVectorizer(stop_words='english', max_features=150)
    X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
    X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
    feature_transform.fit(list(X_train['features']) + list(X_test['features']))

    train_size = len(X_train)
    low_count = len(X_train[X_train['interest_level'] == 0])
    medium_count = len(X_train[X_train['interest_level'] == 1])
    high_count = len(X_train[X_train['interest_level'] == 2])

    managers_with_one_lot = find_objects_with_only_one_record('manager_id')
    buildings_with_one_lot = find_objects_with_only_one_record('building_id')
    addresses_with_one_lot = find_objects_with_only_one_record('display_address')

    print("Starting transformations")        
    X_train = transform_data(X_train)    
    X_test = transform_data(X_test) 
    y = X_train['interest_level'].ravel()

    print("Normalizing high cardinality data...")
    normalize_high_cardinality_data()
    transform_categorical_data()
    remove_columns(X_train)
    remove_columns(X_test)
    
    features = X_train.columns
    X_train.replace(np.inf, np.nan, inplace=True)
    X_test.replace(np.inf, np.nan, inplace=True)
    X_train.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)
    ss =StandardScaler()
    alldata = pd.concat([X_train[features],X_test[features]])
    ss.fit(alldata)
    scaledtrain = X_train.copy()
    scaledtest = X_test.copy()
    scaledtrain[features] = ss.transform(scaledtrain[features])
    scaledtrain['interest_level'] = y
    scaledtest[features] = ss.transform(scaledtest[features])
   
    totalpredictions = np.zeros((scaledtrain.shape[0],3))
    trainmediumhighpreds = GPNotLo(scaledtrain)
    trainhighpreds = GPHighNotMedium(scaledtrain)
    totalpredictions[:, 0] = 1.0-trainmediumhighpreds
    totalpredictions[:, 1] = trainmediumhighpreds*(1.0-trainhighpreds)
    totalpredictions[:, 2] = trainmediumhighpreds*trainhighpreds
    print(log_loss(scaledtrain.interest_level.values,totalpredictions))
    
    testtotalpredictions = np.zeros((scaledtest.shape[0], 3))
    testmediumhighpreds = GPNotLo(scaledtest)
    testhighpreds = GPHighNotMedium(scaledtest)
    testtotalpredictions[:, 0] = 1.0-testmediumhighpreds
    testtotalpredictions[:, 1] = testmediumhighpreds*(1.0-testhighpreds)
    testtotalpredictions[:, 2] = testmediumhighpreds*testhighpreds
    sub = pd.DataFrame(data = {'listing_id': X_test['listing_id'].ravel()})
    sub['low'] = testtotalpredictions[:, 0]
    sub['medium'] = testtotalpredictions[:, 1]
    sub['high'] = testtotalpredictions[:, 2]
    sub.to_csv("submission.csv", index = False, header = True)
