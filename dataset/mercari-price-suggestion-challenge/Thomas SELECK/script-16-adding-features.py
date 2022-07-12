###############################################################################
# First solution for the Mercari Price Suggestion Challenge                   #
#                                                                             #
# This is the entry point of the solution.                                    #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-01-01                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import sys
import wordbatch
import string

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.random import check_random_state
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.csr import csr_matrix
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
import re
from nltk.corpus import stopwords
import gc

from wordbatch import WordBatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

def load_data(training_set_path_str, testing_set_path_str, enable_validation, target_name_str, validation_type = "random"):
    """
    This function is a wrapper for the loading of the data.

    Parameters
    ----------
    training_set_path_str : string
            A string containing the path of the training set file.

    testing_set_path_str : string
            A string containing the path of the testing set file.

    enable_validation : boolean
            A boolean indicating if we are validating our model or if we are creating a submission for Kaggle.

    target_name_str : string
            A string indicating the target column name.

    validation_type : string
            A string indicating either if the validation split is random or time based.

    Returns
    -------
    training_set_df : pd.DataFrame
            A pandas DataFrame containing the training set.

    testing_set_df : pd.DataFrame
            A pandas DataFrame containing the testing set.
            
    target_sr : pd.Series
            The target values for the training part.

    truth_sr : pd.Series
            The target values for the validation part.
    """

    # Load the data
    print("Loading the data...")
    training_set_df = pd.read_csv(training_set_path_str, sep = "\t", index_col = 0)
    testing_set_df = pd.read_csv(testing_set_path_str, sep = "\t", index_col = 0)

    # Remove rows from training set where the price is zero
    print("Removing rows from training set where the target is zero...")
    training_set_df = training_set_df.loc[training_set_df[target_name_str] > 0]
    
    # Generate a validation set if enable_validation is True
    if enable_validation:
        print("Generating validation set...")
        test_size_ratio = 0.2
        if type == "random":
            X_train, X_test = train_test_split(training_set_df, test_size = test_size_ratio, random_state = 2017)
            training_set_df = pd.DataFrame(X_train, columns = training_set_df.columns)
            testing_set_df = pd.DataFrame(X_test, columns = training_set_df.columns)
        else:
            split_threshold = int(training_set_df.shape[0] * (1 - test_size_ratio))
            testing_set_df = training_set_df.iloc[split_threshold:]
            training_set_df = training_set_df.iloc[0:split_threshold]
    
        # Extract truth / target
        truth_sr = testing_set_df[target_name_str]
        testing_set_df = testing_set_df.drop(target_name_str, axis = 1)

        # Reindex DataFrames
        training_set_df = training_set_df.reset_index(drop = True)
        testing_set_df = testing_set_df.reset_index(drop = True)
        truth_sr = truth_sr.reset_index(drop = True)

        print("Generating validation set... done")
    else:
        truth_sr = None

    # Extract target for training set
    target_sr = training_set_df[target_name_str]
    training_set_df = training_set_df.drop(target_name_str, axis = 1)

    print("Loading data... done")

    return training_set_df, testing_set_df, target_sr, truth_sr

class LGBMWrapper(BaseEstimator):
    """
    The purpose of this class is to provide a wrapper for a LightGBM model, with cross-validation for finding the best number of rounds.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = -1, random_state = 0, test_size = 0.1):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the XGBoost model.

        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function XGBoost will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for XGBoost training.

        random_state : zero or positive integer
                Seed used by XGBoost to ensure reproductibility.

        test_size : float between 0 and 1.
                This indicates the size of the test set.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.__params = params
        self.__early_stopping_rounds = early_stopping_rounds
        self.__custom_eval_function = custom_eval_function
        self.__maximize = maximize
        self.__nrounds = nrounds
        self.__random_state = random_state
        self.__test_size = test_size
        self.__params["seed"] = self.__random_state
        self.__lgb_model = None
        self.__model_name = "LightGBM"

    def get_model_name(self):
        return self.__model_name

    def get_LGB_model(self):
        return self.__lgb_model

    def get_nb_rounds(self):
        return self.__nrounds

    def get_params(self):
        return self.__params

    def PrintFeatureImportance(self, maxCount = -1):
        importance = self.__lgb_model.get_score(importance_type = "gain")
        importance = sorted(importance.items(), key = itemgetter(1), reverse = True)

        for i, item in enumerate(importance):
            if i < maxCount or maxCount < 0:
                print(i + 1, "/", len(importance), ":", item[0])

    def plot_features_importance(self, importance_type = "gain"):
        lgb.plot_importance(self.__lgb_model, importance_type = importance_type)
        plt.show()

    def get_features_importance(self, importance_type = "gain"):
        importance = self.__lgb_model.feature_importance(importance_type = "gain")
        features_names = self.__lgb_model.feature_name()

        return pd.DataFrame({"feature": features_names, "importance": importance}).sort_values(by = "importance", ascending = False).reset_index(drop = True)

    def fit(self, X, y):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """

        print("Preparing data for LightGBM...")
        #columns_names_lst = X[1]#[0:48663]
        #X = X[0]
                       
        if self.__nrounds == -1:
            dtrain = lgb.Dataset(X, label = y)
            watchlist = [dtrain]

            print("    Cross-validating LightGBM classifier with seed: " + str(self.__random_state) + "...")
            cv_output = lgb.cv(self.__params, dtrain, num_boost_round = 10000, early_stopping_rounds = self.__early_stopping_rounds, show_stdv = True)

            self.__nrounds = cv_output.shape[0]

            print("    Training LightGBM classifier with seed: " + str(self.__random_state) + " and num rounds = " + str(self.__nrounds) + "...")
            self.__lgb_model = lgb.train(self.__params, dtrain, self.__nrounds, watchlist, early_stopping_rounds = self.__early_stopping_rounds, verbose_eval = 100)
        else:
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = self.__test_size, random_state = self.__random_state)
            
            print("Training LightGBM...")
            dtrain = lgb.Dataset(X_train, label = y_train)#, feature_name = columns_names_lst)
            dvalid = lgb.Dataset(X_eval, label = y_eval)#, feature_name = columns_names_lst)
            watchlist = [dtrain, dvalid]

            self.__lgb_model = lgb.train(self.__params, dtrain, self.__nrounds, watchlist, early_stopping_rounds = self.__early_stopping_rounds, verbose_eval = 100)

        return self

    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.__lgb_model is None:
            raise ValueError("You MUST train the XGBoost model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.__lgb_model.predict(X)

        return predictions_npa

class PreprocessingStep2(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.__num_brands = 4000
        self.__num_categories = 1000
        self.__min_df = 10
        self.__max_features_item_description = 50000
        
        self.__numbers_re = re.compile("[0-9]+")
        self.__units_re = re.compile("[a-z]+")
        self.__memory_re = re.compile("([0-9]+\s?tb(\s|$))|([0-9]+\s?gb(\s|$))|([0-9]+\s?mb(\s|$))")
        self.__memory_gb_re = re.compile("(([0-9]+)(\s?gb\s|$))")
                        
    def clean_text(self, x):
        punctuation_re = re.compile("[" + re.escape(string.punctuation) + "\\r\\t\\n]")
        garbage_re = re.compile("[\W_]+")
        spaces_re = re.compile("\s+")

        x = punctuation_re.sub(" ", x)
        x = garbage_re.sub(" ", x)
        x = spaces_re.sub(" ", x)

        return x

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        # Compute the mean target for each category found in 'category_name'
        tmp_df = X.copy()
        tmp_df["target"] = y
        tmp_df["category_name"].fillna("no_category_name", inplace = True)
        tmp2_df = tmp_df[["category_name", "target"]].groupby("category_name").mean()
        self.__mean_target_by_category_dict = tmp2_df.to_dict()["target"]

        # For categories that have less than 10 samples, group them and replace their mean by the group mean
        categories_count_df = X["category_name"].value_counts()
        least_occuring_categories_lst = categories_count_df.loc[categories_count_df < 10].index.tolist()
        mean_value = tmp_df["target"].loc[tmp_df["category_name"].isin(least_occuring_categories_lst)].mean()

        for category in categories_count_df.loc[categories_count_df < 10].index:
            self.__mean_target_by_category_dict[category] = mean_value

        ## Get the name of the category level
        X["category_1"] = X["category_name"].map(lambda x: x.split("/")[0] if type(x) == str and len(x.split("/")) > 0 else None)

        ## Fill missing values
        X["category_1"].fillna("missing", inplace = True)

        # Group least occuring categories
        tmp = X["category_1"].value_counts()
        tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_categories]
        X.loc[~X["category_1"].isin(tmp), "category_1"] = "others"

        self.__brands_groups = X[["brand_name", "category_1"]].groupby(["category_1", "brand_name"]).size().reset_index()
            
        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """

        start_time = time.time()

        # Add indicator for missing brands
        X["is_no_name"] = (X["brand_name"].isnull()).astype(np.int8)

        # Reverse the value of the 'item_condition_id' feature to make increasing value wrt to the price (remember, 1 -> new item, 5 -> broken thing)
        X["reversed_item_condition_id"] = 5 - X["item_condition_id"]

        # Add mean target by category
        X["category_name_mean_target"] = round(X["category_name"].map(self.__mean_target_by_category_dict), 0)
        X["category_name_mean_target"].fillna(self.__mean_target_by_category_dict["no_category_name"], inplace = True)
        
        # Bin mean price by category_name
        X["category_name_target_mean_lt10"] = (X["category_name_mean_target"] < 10).astype(np.int8)
        X["category_name_target_mean_10_20"] = ((X["category_name_mean_target"] >= 10) & (X["category_name_mean_target"] < 20)).astype(np.int8)
        X["category_name_target_mean_lt20"] = (X["category_name_mean_target"] < 20).astype(np.int8)
        X["category_name_target_mean_20_40"] = ((X["category_name_mean_target"] >= 20) & (X["category_name_mean_target"] < 40)).astype(np.int8)
        X["category_name_target_mean_40_50"] = ((X["category_name_mean_target"] >= 40) & (X["category_name_mean_target"] < 50)).astype(np.int8)
        X["category_name_target_mean_50_75"] = ((X["category_name_mean_target"] >= 50) & (X["category_name_mean_target"] < 75)).astype(np.int8)
        X["category_name_target_mean_gt75"] = (X["category_name_mean_target"] >= 75).astype(np.int8)
        X["binned_category_name_target_mean"] = X["category_name_target_mean_lt20"] + 2 * X["category_name_target_mean_20_40"] + 3 * X["category_name_target_mean_40_50"] + 4 * X["category_name_target_mean_50_75"] + 5 * X["category_name_target_mean_gt75"]

        # Add indicator for no description
        #X["item_description_is_null"] = (X["item_description"].isnull()).astype(np.int8)

        # Extract categories hierarchy from 'category_name' feature
        for i in range(3): #5
            ## Get the name of the category level
            X["category_" + str(i + 1)] = X["category_name"].map(lambda x: x.split("/")[i] if type(x) == str and len(x.split("/")) > i else None)

            ## Fill missing values
            X["category_" + str(i + 1)].fillna("missing", inplace = True)

        X.drop("category_name", axis = 1, inplace = True)
        
        # Fill missing values
        X["brand_name"].fillna(value = "missing", inplace = True)
        X["name"].fillna(value = "missing", inplace = True)
        X["item_description"].fillna(value = "missing", inplace = True)

        # Look for luxury brands: they are expensive
        luxury_brands_set = {"MCM", "MCM Worldwide", "Louis Vuitton", "Burberry", "Burberry London", "Burberry Brit", "HERMES", "Tieks", "Rolex", "Apple", "Gucci", "Valentino", 
                             "Valentino Garavani", "RED Valentino", "Cartier", "Christian Louboutin", "Yves Saint Laurent", "Saint Laurent", "YSL Yves Saint Laurent", "Georgio Armani",
                             "Armani Collezioni", "Emporio Armani"}

        X["is_luxury_brand"] = X["brand_name"].apply(lambda x: int(x in luxury_brands_set))
                
        # Look for some important keywords 
        X["joint_description"] = X["name"].str.cat(X["item_description"], sep = " ").str.lower()
        X["contains_dust"] = X["joint_description"].str.count("dust", flags = re.IGNORECASE)
        X["contains_gold"] = X["joint_description"].str.count("gold", flags = re.IGNORECASE)
        #X["contains_silver"] = X["joint_description"].str.contains("silver", case = False).astype(np.int8)
        #X["contains_leather"] = X["joint_description"].str.contains("leather", case = False).astype(np.int8)
        X["name_contains_lularoe"] = X["name"].str.contains("lularoe", case = False).astype(np.int8)
        X["name_contains_bundle"] = X["name"].str.count("bundle", flags = re.IGNORECASE)
        #X["name_contains_iphone"] = X["name"].str.contains("iphone", case = False).astype(np.int8)
        #X["name_contains_nike"] = X["name"].str.contains("nike", case = False).astype(np.int8)
        #X["name_contains_rare"] = X["name"].str.contains("rare", case = False).astype(np.int8)
        #X["name_contains_charger"] = X["name"].str.contains("charger", case = False).astype(np.int8)
        #X["item_description_contains_authentic"] = X["item_description"].str.contains("authentic", case = False).astype(np.int8)
        #X["item_description_contains_box"] = X["item_description"].str.contains("box", case = False).astype(np.int8)

        # Generate feature based on gold purity
        """X["contains_10k"] = X["joint_description"].str.contains("10k", case = False).astype(np.int8)
        X["contains_14k"] = X["joint_description"].str.contains("14k", case = False).astype(np.int8)
        X["contains_18k"] = X["joint_description"].str.contains("18k", case = False).astype(np.int8)
        X["contains_22k"] = X["joint_description"].str.contains("22k", case = False).astype(np.int8)
        X["contains_24k"] = X["joint_description"].str.contains("24k", case = False).astype(np.int8)
        X["gold_purity"] = (10 / 24) * X["contains_10k"] + (14 / 24) * X["contains_14k"] + (18 / 24) * X["contains_18k"] + (22 / 24) * X["contains_22k"] + X["contains_24k"]

        # Generate features based on flash memory size
        X["contains_8gb"] = X["joint_description"].str.contains("8gb|8 gb", case = False).astype(np.int8)
        X["contains_16gb"] = X["joint_description"].str.contains("16gb|16 gb", case = False).astype(np.int8)
        X["contains_32gb"] = X["joint_description"].str.contains("32gb|32 gb", case = False).astype(np.int8)
        X["contains_64gb"] = X["joint_description"].str.contains("64gb|64 gb", case = False).astype(np.int8)
        X["contains_128gb"] = X["joint_description"].str.contains("128gb|128 gb", case = False).astype(np.int8)
        X["flash_memory_size"] = 8 * X["contains_32gb"] + 16 * X["contains_64gb"] + 32 * X["contains_32gb"] + 64 * X["contains_64gb"] + 128 * X["contains_128gb"]"""
                
        # Group least occuring brands
        tmp = X["brand_name"].value_counts()
        tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_brands]
        X.loc[~X["brand_name"].isin(tmp), "brand_name"] = "missing"

        # Group least occuring categories
        for i in range(3):
            tmp = X["category_" + str(i + 1)].value_counts()
            tmp = tmp.loc[tmp.index != "missing"].index[:self.__num_categories]
            X.loc[~X["category_" + str(i + 1)].isin(tmp), "category_" + str(i + 1)] = "others"

        print("[{}] Cut completed.".format(time.time() - start_time))
        
        # Convert 'item_condition_id' to dummies
        X = pd.concat([X, pd.get_dummies(X["item_condition_id"], prefix = "item_condition_id")], axis = 1)
        print('[{}] Get dummies on `item_condition_id` completed.'.format(time.time() - start_time))

        # Compute some statistics on 'name' and 'item_description'
        X["name_nb_chars"] = X["name"].str.len()
        X["item_description_nb_chars"] = X["item_description"].str.len()
        X["nb_chars"] = X["name_nb_chars"] + X["item_description_nb_chars"]
        
        X["name_nb_tokens"] = X["name"].str.lower().str.split(" ").str.len()
        X["item_description_nb_tokens"] = X["item_description"].str.lower().str.split(" ").str.len()
        X["nb_tokens"] = X["name_nb_tokens"] + X["item_description_nb_tokens"]
        X["tokens_ratio"] = X["name_nb_tokens"] / X["item_description_nb_tokens"]

        X["name_nb_words"] = X["name"].str.count("(\s|^)[a-z]+(\s|$)")
        X["item_description_nb_words"] = X["item_description"].str.count("(\s|^)[a-z]+(\s|$)")
        X["nb_words"] = X["name_nb_words"] + X["item_description_nb_words"]

        X["name_nb_numbers"] = X["name"].str.count("(\s|^)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(\s|$)")
        X["item_description_nb_numbers"] = X["item_description"].str.count("(\s|^)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(\s|$)")
        X["nb_numbers"] = X["name_nb_numbers"] + X["item_description_nb_numbers"]

        X["name_nb_letters"] = X["name"].str.count("[a-zA-Z]")
        X["item_description_nb_letters"] = X["item_description"].str.count("[a-zA-Z]")
        X["nb_letters"] = X["name_nb_letters"] + X["item_description_nb_letters"]

        X["name_nb_digits"] = X["name"].str.count("[0-9]")
        X["item_description_nb_digits"] = X["item_description"].str.count("[0-9]")
        X["nb_digits"] = X["name_nb_digits"] + X["item_description_nb_digits"]

        for group in ["Beauty", "Electronics", "Handmade", "Home", "Other", "Sports & Outdoors"]:
            brands_lst = self.__brands_groups["brand_name"].loc[self.__brands_groups["category_1"] == group].tolist()
            X["brand_group_" + group] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        brands_lst = ["Michael Kors", "Louis Vuitton", "Lululemon", "LuLaRoe", "Kendra Scott", "Tory Burch", "Apple", "Kate Spade", "UGG Australia", "Coach", "Gucci", "Rae Dunn", "Tiffany & Co.",
                      "Rock Revival", "Adidas", "Beats", "Burberry", "Christian Louboutin", "David Yurman", "Ray-Ban", "Chanel"]
        X["is_most_expensive"] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        brands_lst = ["FOREVER 21", "Old Navy", "Carter's", "Elmers", "NYX", "Maybelline", "Disney", "American Eagle", "PopSockets", "Wet n Wild", "Hollister", "Pokemon", "Hot Topic", "Konami", 
                      "Charlotte Russe", "H&M", "e.l.f.", "Bath & Body Works", "Gap"]
        X["is_cheapest"] = (X["brand_name"].isin(brands_lst)).astype(np.int8)

        """X["memory_unit"] = X["joint_description"].str.count(self.__memory_re)
        tmp_df = X["joint_description"].str.extractall(self.__memory_gb_re).reset_index()
        tmp_df.columns = ["train_id", "match", "0", "measure_unit_gb", "2"]
        tmp_df.dropna(inplace = True)
        tmp_df = tmp_df[["train_id", "measure_unit_gb"]]
        tmp_df["measure_unit_gb"] = tmp_df["measure_unit_gb"].astype(np.int32)
        tmp_df = tmp_df.groupby("train_id").sum()
        X = X.merge(tmp_df, how = "left", left_index = True, right_index = True)
        X["measure_unit_gb"].fillna(0, inplace = True)
        X["measure_unit_gb"] = X["measure_unit_gb"].astype(np.float32)"""
        X.drop(["joint_description"], axis = 1, inplace = True)
        
        # Clean the text for 'name' and 'item_description'
        """X["name"] = X["name"].apply(self.clean_text)
        X["item_description"] = X["item_description"].apply(self.clean_text)"""
        
        return X

class SparseTextEncoder(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a text encoder using bag of words or TF-IDF representation.
    """

    def __init__(self, columns_names_lst, encoders_lst):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                This contains the names of the columns we want to transform.

        encoders_lst : list
                This contains the encoders chosen for each column of the columns_names_lst list.
                                
        Returns
        -------
        None
        """

        self.__columns_names_lst = columns_names_lst
        self.__min_df = 10
        self.__encoders_lst = encoders_lst
        self.__encoders_masks_lst = [None for i in encoders_lst]

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        print("Method not implemented! Please call fit_transform() instead.")

        return self

    def fit_transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data and then transform the associated data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.__columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]

        print("Regular columns shape:", arrays_lst[0].shape)
        print("Regular columns names length:", len(columns_names_lst))

        nnz_threshold = 2

        for idx, column in enumerate(self.__columns_names_lst):
            X[column].fillna("NaN", inplace = True)
            
            if type(self.__encoders_lst[idx]) == CountVectorizer or type(self.__encoders_lst[idx]) == TfidfVectorizer:
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_" + w for w in self.__encoders_lst[idx].get_feature_names()]
            elif type(self.__encoders_lst[idx]) == LabelBinarizer:
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoder_features_csr.shape[1])]
            elif type(self.__encoders_lst[idx]) == wordbatch.WordBatch:
                self.__encoders_lst[idx].dictionary_freeze = True
                encoder_features_csr = self.__encoders_lst[idx].fit_transform(X[column])
                self.__encoders_masks_lst[idx] = np.array(np.clip(encoder_features_csr.getnnz(axis = 0) - nnz_threshold, 0, 1), dtype = bool)
                encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

                encoder_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoder_features_csr.shape[1])]

            print(column, ": shape:", encoder_features_csr.shape)
            print(column, ": columns names length:", len(encoder_columns_names_lst))

            arrays_lst.append(encoder_features_csr)
            columns_names_lst.extend(encoder_columns_names_lst)

        sparse_merge = hstack(arrays_lst).tocsr()

        print("sparse_merge: shape:", sparse_merge.shape)
        
        print("*** Sparse text encoder: transform in ", round(time.time() - start_time, 3), "seconds ***")

        #return (sparse_merge, columns_names_lst)
        return sparse_merge

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.__columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        
        for idx, column in enumerate(self.__columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            encoder_features_csr = self.__encoders_lst[idx].transform(X[column])
            encoder_features_csr = encoder_features_csr[:, self.__encoders_masks_lst[idx]]

            arrays_lst.append(encoder_features_csr)
            
        sparse_merge = hstack(arrays_lst).tocsr()
        
        return sparse_merge

class BaggingEstimator(BaseEstimator):
    """
    The purpose of this class is to provide a wrapper for a bagging estimator with a Scikit-Learn like interface.
    """
    
    def __init__(self, estimators_lst, estimators_weights_lst):
        """
        Class' constructor

        Parameters
        ----------
        estimators_lst : list
                This list contains all the estimators we want to use.

        estimators_weights_lst: list
                This list contains the weights associated with each estimator for the final prediction.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.__estimators_lst = estimators_lst
        self.__estimators_weights_lst = estimators_weights_lst

    def fit(self, X, y):
        """
        This method trains the bagging estimator.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """
            
        print("Training all models...")
        for estimator in self.__estimators_lst:
            estimator.fit(X, y)
                  
        return self

    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        predictions_npa = np.zeros(X.shape[0])

        print("Making predictions on all models...")
        for estimator, weight in zip(self.__estimators_lst, self.__estimators_weights_lst):
            predictions_npa += weight * estimator.predict(X)
            
        return predictions_npa

def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1 and x not in stopwords])
        
# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2017)

    enable_validation = False  

    features_to_be_scaled_lst = ["reversed_item_condition_id", "category_name_mean_target", "binned_category_name_target_mean", "name_nb_chars", "item_description_nb_chars", 
                                 "nb_chars", "name_nb_tokens", "item_description_nb_tokens", "nb_tokens", "tokens_ratio", "name_nb_words", "item_description_nb_words", "nb_words", 
                                 "name_nb_numbers", "item_description_nb_numbers", "nb_numbers", "name_nb_letters", "item_description_nb_letters", "nb_letters", "name_nb_digits", 
                                 "item_description_nb_digits", "nb_digits", "contains_dust", "contains_gold", "name_contains_lularoe", "name_contains_bundle"]

    columns_to_be_encoded_lst = ["name", "item_description", "category_1", "category_2", "category_3", "brand_name", "item_condition_id"]

    coeffs_dict = {"Men": [-0.01180431, 0.6517549, 0.35800552], "Electronics": [-0.02658065, 0.70573716, 0.32172113], "Women": [-0.04936933, 0.65523746, 0.39512383], 
                   "Home": [-0.04675894, 0.72631163, 0.32075199], "Sports & Outdoors": [-0.11175634, 0.80728063, 0.30652454], "Vintage & Collectibles": [-0.06974298, 0.70257194, 0.36689862], 
                   "Beauty": [-0.00074757, 0.67291934, 0.32724406], "Other": [0.03854487, 0.69971427, 0.25985178], "Kids": [0.01197509, 0.63588165, 0.35228066], 
                   "others": [0.14820591, 0.75975425, 0.10707419], "Handmade": [-0.05670657, 0.68721659, 0.37363133]}

    encoders_lst = [WordBatch(normalize_text, extractor = (WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], "hash_size": 2 ** 29, "norm": None, "tf": 'binary', "idf": None}), procs = 8),
                    WordBatch(normalize_text, extractor = (WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0], "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0, "idf": None}), procs = 8),
                    CountVectorizer(),
                    CountVectorizer(),
                    CountVectorizer(),
                    LabelBinarizer(sparse_output = True),
                    LabelBinarizer(sparse_output = True)]

    params = {
        'learning_rate': 0.20,
        'application': 'regression',
        'max_depth': 10,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.75,
        'nthread': 4
    }

    # Load the data
    train, test, target, truth = load_data("../input/train.tsv", "../input/test.tsv", enable_validation, "price") # truth is None
    
    # temporary; don't forget truth
    #if enable_validation:
    """test = test.sample(n = 3600000, replace = True)
    test = test.reset_index()
    test.drop("test_id", axis = 1, inplace = True)"""
    
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
            
    ps = PreprocessingStep2()
    train = ps.fit_transform(train, target)

    sc = StandardScaler()
    train[features_to_be_scaled_lst] = sc.fit_transform(train[features_to_be_scaled_lst])

    ste = SparseTextEncoder(columns_to_be_encoded_lst, encoders_lst)
    X = ste.fit_transform(train, target)

    gc.collect()

    # Take the log of the target
    y = np.log1p(target)

    if enable_validation:
        truth_sr = np.log1p(truth)

    del train, target
    gc.collect()

    FTRL_model = FTRL(alpha = 0.01, beta = 0.1, L1 = 0.00001, L2 = 1.0, D = X.shape[1], iters = 50, inv_link = "identity", threads = 1)
    FTRL_model.fit(X, y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))

    FM_FTRL_model = FM_FTRL(alpha = 0.01, beta = 0.01, L1 = 0.00001, L2 = 0.1, D = X.shape[1], alpha_fm = 0.01, L2_fm = 0.0, init_fm = 0.01, D_fm = 200, e_noise = 0.0001, iters = 17, inv_link = "identity", threads = 4)
    FM_FTRL_model.fit(X, y)
    print('[{}] Train FM FTRL completed'.format(time.time() - start_time))

    # Remove features with document frequency <=100
    print("Before removing features with document frequency <=100:", X.shape)
    mask = np.array(np.clip(X.getnnz(axis = 0) - 100, 0, 1), dtype = bool)
    X = X[:, mask]
    print("After removing features with document frequency <=100:", X.shape)
    
    print('[{}] Generating LightGBM data.'.format(time.time() - start_time))
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.10, random_state = 100) 
    d_train = lgb.Dataset(train_X, label = train_y)
    d_valid = lgb.Dataset(valid_X, label = valid_y)
    watchlist = [d_train, d_valid]

    lgb_model = lgb.train(params, train_set = d_train, num_boost_round = 2200, valid_sets = watchlist, early_stopping_rounds = 1000, verbose_eval = 100) 
    print('[{}] Train LGB completed.'.format(time.time() - start_time))

    del X, y, train_X, valid_X, train_y, valid_y, d_train, d_valid, watchlist
    gc.collect()

    # Test part
    submission = pd.DataFrame({"test_id": test.index.values, "price": np.zeros(test.shape[0])})
    
    ## Get the name of the category level
    test["category_1"] = test["category_name"].map(lambda x: x.split("/")[0] if type(x) == str and len(x.split("/")) > 0 else None)

    ## Fill missing values
    test["category_1"].fillna("missing", inplace = True)

    # Group least occuring categories
    tmp = test["category_1"].value_counts()
    tmp = tmp.loc[tmp.index != "missing"].index[:1000]
    test.loc[~test["category_1"].isin(tmp), "category_1"] = "others"

    for category in test["category_1"].unique():
        tmp = ps.transform(test.loc[test["category_1"] == category])
        tmp[features_to_be_scaled_lst] = sc.transform(tmp[features_to_be_scaled_lst])
        X_test = ste.transform(tmp)
    
        gc.collect()

        predsF = FTRL_model.predict(X_test)
        print("predsF shape:", predsF.shape)
        print("predsF NaNs:", np.isnan(predsF).sum())
        print('[{}] Predict FTRL completed'.format(time.time() - start_time))

        predsFM = FM_FTRL_model.predict(X_test)
        print("predsFM shape:", predsFM.shape)
        print("predsFM NaNs:", np.isnan(predsFM).sum())
        print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

        X_test = X_test[:, mask]

        predsL = lgb_model.predict(X_test)
        print('[{}] Predict LGB completed.'.format(time.time() - start_time))

        preds = coeffs_dict[category][0] * predsF + coeffs_dict[category][1] * predsFM + coeffs_dict[category][2] * predsL
        submission["price"].loc[test["category_1"] == category] = np.expm1(preds)

        del X_test, tmp
        gc.collect()
        
    # Prevent issues with RMSLE
    submission["price"] = np.abs(submission["price"])
    
    submission.to_csv("submission.csv", index = False)
    
    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")