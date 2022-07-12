###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed for selecting the best subset of features#
# from a set of features. It is compatible with the Scikit-Learn framework.   #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-02-23                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time
import warnings
import networkx as nx
import multiprocessing as mp
import gc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler

class DuplicatedFeaturesRemover(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that removes features that are duplicated.
    """

    def __init__(self, ignored_features_lst = [], n_jobs = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        ignored_features_lst : list (default = [])
                This list contains the name of all features that will be ignored by the detection of duplicates.

        n_jobs : integer (default = -1)
                This indicates the number of CPU cores to use to do the processing. If -1, all cores are used.
                                                
        Returns
        -------
        None
        """

        self.ignored_features_lst = ignored_features_lst
        self._duplicated_features_lst = []

        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

        manager = mp.Manager()
        self.edges_lst = manager.list()
        self._constant_features_lst = manager.list()
        self.ns = manager.Namespace()

    def _single_thread_worker(self, X):
        for i in range(len(self._features_lst)):
            if i % 10 == 0:
                print("Processing feature", i, "/", len(self._features_lst))

            for j in range(i + 1, len(self._features_lst)):
                f1 = self._features_lst[features_tuple[i]]
                f2 = self._features_lst[features_tuple[j]]

                if X[f1].nunique() == 1 or X[f1].isnull().all():
                    if f1 not in self._constant_features_lst:
                        self._constant_features_lst.append(f1)
                        print("    -", f1, "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                        continue

                if X[f2].nunique() == 1 or X[f2].isnull().all():
                    if f2 not in self._constant_features_lst:
                        self._constant_features_lst.append(f2)
                        print("    -", f2, "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                        continue

                # If both features doesn't have the same number of levels, then they aren't duplicated
                f1_nb_levels = X[f1].nunique()
                f2_nb_levels = X[f2].nunique()

                if f1_nb_levels == f2_nb_levels:
                    try: # For mixed type columns (containing numbers and strings), Pandas crosstab can fail.
                        confusion_matrix_df = pd.crosstab(X[f1], X[f2], normalize = "index")
                    except:
                        X[f1] = X[f1].astype(str)
                        X[f2] = X[f2].astype(str)

                        confusion_matrix_df = pd.crosstab(X[f1], X[f2], normalize = "index")
                    
                    # Add an edge in the graph indicating that both features are duplicated
                    if confusion_matrix_df.shape[0] == confusion_matrix_df.shape[1] and np.count_nonzero(confusion_matrix_df) == confusion_matrix_df.shape[0] and f1 not in self._duplicated_features_lst:
                        self.edges_lst += [(f1, f2)]

    def _worker(self, q, iolock):

        # Get dataset
        X = self.ns.dataset

        while True:
            features_tuple = q.get()

            if features_tuple is None:
                break
            else:
                f1 = self._features_lst[features_tuple[0]]
                f2 = self._features_lst[features_tuple[1]]

            if X[f1].nunique() == 1 or X[f1].isnull().all():
                if f1 not in self._constant_features_lst:
                    self._constant_features_lst.append(f1)
                    print("    -", f1, "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                continue

            if X[f2].nunique() == 1 or X[f2].isnull().all():
                if f2 not in self._constant_features_lst:
                    self._constant_features_lst.append(f2)
                    print("    -", f2, "is constant. Please use features_selectors.ConstantFeaturesRemover to remove it!")
                continue

            # If both features doesn't have the same number of levels, then they aren't duplicated
            f1_nb_levels = X[f1].nunique()
            f2_nb_levels = X[f2].nunique()

            if f1_nb_levels == f2_nb_levels:
                try: # For mixed type columns (containing numbers and strings), Pandas crosstab can fail.
                    confusion_matrix_df = pd.crosstab(X[f1], X[f2], normalize = "index")
                except:
                    X[f1] = X[f1].astype(str)
                    X[f2] = X[f2].astype(str)

                    confusion_matrix_df = pd.crosstab(X[f1], X[f2], normalize = "index")
                    
                # Add an edge in the graph indicating that both features are duplicated
                if confusion_matrix_df.shape[0] == confusion_matrix_df.shape[1] and np.count_nonzero(confusion_matrix_df) == confusion_matrix_df.shape[0] and f1 not in self._duplicated_features_lst:
                    iolock.acquire()
                    self.edges_lst += [(f1, f2)]
                    iolock.release()
        
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
        self: TextStatisticsGenerator object
                Return current object.
        """

        print("Removing duplicated features...")

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)
        
        # Separate numeric columns from categorical ones
        X_num = X_copy.select_dtypes(include = np.number)
        X_cat = X_copy.select_dtypes(include = "object")

        # For numerical features, compute correlation
        if X_num.shape[1] > 0:
            cormat_df = X_num.corr()

            for i in range(cormat_df.shape[0]):
                for j in range(i + 1, cormat_df.shape[0]):
                    if cormat_df.iloc[i, j] == 1:
                        self.edges_lst += [(cormat_df.columns[i], cormat_df.columns[j])]

        if X_cat.shape[1] > 0:
            self._features_lst = list(set(X_cat.columns.tolist()) - set(self.ignored_features_lst))
            self.ns.dataset = X_cat
            q = mp.Queue(maxsize = self.n_jobs)
            iolock = mp.Lock()

            if self.n_jobs > 1:
                pool = mp.Pool(self.n_jobs, initializer = self._worker, initargs = (q, iolock))

                for i in range(len(self._features_lst)):
                    if i % 10 == 0:
                        print("Processing feature", i, "/", len(self._features_lst))

                    for j in range(i + 1, len(self._features_lst)):
                        q.put((i, j))  # blocks until q below its max size

                # tell workers we're done
                for _ in range(self.n_jobs):  
                    q.put(None)

                pool.close()
                pool.join()
            else:
                self._single_thread_worker(X_cat)
            
        # Use graph to detect duplicates
        G = nx.Graph()
        
        # Construct the graph  
        G.add_edges_from(list(self.edges_lst))

        # Get all connected components
        connected_components_lst = list(nx.connected_components(G))

        # For each connected component, if it's a complete graph, then features given by the nodes are duplicated
        for connected_component in connected_components_lst:
            connected_component = list(connected_component)

            # Get degree of each node
            nodes_degrees_set = list(set(dict(G.degree(connected_component)).values()))

            # If the subgraph is complete
            if len(nodes_degrees_set) == 1 and nodes_degrees_set[0] == len(connected_component) - 1:
                print("    - Feature:", connected_component[0], "is duplicated, the duplicates are:", ", ".join(connected_component[1:]))
                self._duplicated_features_lst.extend(connected_component[1:])

        if len(self._duplicated_features_lst) > 0:
            print("\nDuplicated features that will be removed:")
            for f in self._duplicated_features_lst:
                print("    -", f)
        else:
            print("    No duplicated feature found!")

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
                Transformed data.
        """
        
        # Remove features that only have one unique value
        X.drop(self._duplicated_features_lst, axis = 1, inplace = True)

        return X

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains some classes that performs categorical encoding that are #
# compatible with scikit-learn API.                                           #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-07                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import time
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a wrapper for the Scikit-Learn LabelEncoder class, that can manage
    cases where testing set contains new labels that doesn't exist in training set.
    """

    def __init__(self, mapping_dict = None, missing_value_replacement = "NA"):
        """
        This is the class' constructor.

        Parameters
        ----------
        mapping_dict : dictionary
                Use this to provide a custom mapping between current feature levels
                and the integers you want to associate with them.

        missing_value_replacement : string
                Value used to replace missing values.
                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.mapping_dict = mapping_dict
        self.missing_value_replacement = missing_value_replacement

        self._label_encoder = LabelEncoder()
        self._unique_labels_lst = []

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        if self.mapping_dict is None:
            # Fit the LabelEncoder
            self._label_encoder.fit(X.fillna(self.missing_value_replacement))

            # Save the unique labels available in the training set
            self._unique_labels_lst = list(self._label_encoder.classes_)
        else:
            # Save the unique labels available in the mapping dict
            self._unique_labels_lst = list(self.mapping_dict.keys())

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                This is a series containing the data that will be transformed.
        """
        
        # Get new labels (labels not seen in fit)
        column_labels_lst = X.unique()
        unseen_labels_lst = list(set(column_labels_lst) - set(self._unique_labels_lst))

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)
        X_copy.fillna(self.missing_value_replacement, inplace = True)

        # Transform the feature
        if self.mapping_dict is None:
            X_copy.loc[~X_copy.isin(unseen_labels_lst)] = self._label_encoder.transform(X_copy.loc[~X_copy.isin(unseen_labels_lst)])
        else:
            X_copy.loc[~X_copy.isin(unseen_labels_lst)] = X_copy.loc[~X_copy.isin(unseen_labels_lst)].map(self.mapping_dict)

        X_copy.loc[X_copy.isin(unseen_labels_lst)] = np.nan
        
        # If there is no NaNs in the series, cast it to np.int32
        if X_copy.isnull().sum() == 0:
            X_copy = X_copy.astype(np.int32)
            
        return X_copy

class GroupingEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a categorical feature encoder especially designed to
    handle features with high cardinality by grouping scarcest levels into a "OTHER" label. Then,
    it performs one-hot encoding on remaining levels.
    """

    def __init__(self, encoder, threshold, grouping_name = "OTHER"):
        """
        This is the class' constructor.

        Parameters
        ----------
        encoder : scikit-learn transformer
                This is the encoder that will be used to encode the feature after
                grouping its least frequent levels.

        threshold : either integer >= 2 or float between 0 and 1
                - If this is a float between 0 and 1, then only the scarcest levels that cumulated sum represents
                  1 - threshold will be grouped in a class named 'grouping_name'.
                - If this is an integer, then only the 'threshold' most represented levels will be kept. Others will be grouped
                  in a level named 'grouping_name'.

        grouping_name : string (default = "OTHER")
                Name given to the levels grouped for the feature.
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.encoder = encoder
        self.threshold = threshold
        self.grouping_name = grouping_name

        self._kept_levels = []
        self.classes_ = None

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        # Get the names of the levels that will be grouped together
        levels_count_sr = X.value_counts()

        if type(self.threshold) == int and self.threshold >= 2:
            self._kept_levels = levels_count_sr.head(self.threshold).index.tolist()
        elif type(self.threshold) == float and self.threshold > 0 and self.threshold < 1:
            levels_count_df = levels_count_sr.reset_index()
            levels_count_df.columns = ["level", "count"]
            levels_count_df["cumsum"] = levels_count_df["count"].cumsum()
            levels_count_df = levels_count_df.loc[levels_count_df["cumsum"] < self.threshold * levels_count_df["count"].sum()]
            self._kept_levels = levels_count_df["level"].tolist()

        # Fit the encoder
        tmp_sr = X.copy(deep = True)
        tmp_sr.loc[~X.isin(self._kept_levels)] = self.grouping_name

        if isinstance(self.encoder, LeaveOneOutEncoder) or isinstance(self.encoder, TargetAvgEncoder):
            self.encoder.fit(tmp_sr, y)
        else:
            self.encoder.fit(tmp_sr)
        
        self.classes_ = self.encoder.classes_

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
         : numpy array
                Numpy array containing the transformed data.
        """
        
        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Group labels that need to be grouped
        X_copy.loc[~X_copy.isin(self._kept_levels)] = self.grouping_name

        # Apply the transformer
        return self.encoder.transform(X_copy)
    
class TargetAvgEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a transformer that encodes categorical variable by
    replacing each level by its target mean.

    Beware: Only use this transformer with low cardinality features. Otherwise, high overfiting 
    can be introduced in the model. In this case, the LeaveOneOut encoder is preferred.
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

        # Class' attributes
        self.classes_ = None

    def fit(self, X, y):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        # Compute the average target value for each feature level
        self._means_dict = y.groupby(X).mean().to_dict()

        # Compute the global mean of the feature
        self._target_mean = y.mean()

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                Series containing the transformed data.
        """
        
        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Map mean values to feature levels
        X_copy = X_copy.map(self._means_dict)

        # Replace new levels by whole target mean
        X_copy.fillna(self._target_mean, inplace = True)

        return X_copy

class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a leave-one-out coding for categorical features.

    References
    ----------

    .. [1] Strategies to encode categorical variables with many categories. From
    https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154
    """

    def __init__(self, add_gaussian_noise = True, sigma = 0.05):
        """
        This is the class' constructor.

        Parameters
        ----------
        add_gaussian_noise : bool (default = True)
                Add Gaussian noise or not to the data encoded in fit_transform() method. The purpose of this
                is to reduce overfitting.
        
        sigma : float (default = 0.05)
                Standard deviation of the above mentionned Gaussian distribution.
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.add_gaussian_noise = add_gaussian_noise
        self.sigma = sigma

        self.classes_ = None
        
    def fit(self, X, y):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series
                This is the target associated with the X data.

        Returns
        -------
        self: SparseTextEncoder object
                Return current object.
        """

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Compute the global target mean
        self._target_mean = y.mean()

        # Compute the average target value for each feature level
        self._feature_statistics_df = y.groupby(X_copy).agg(["sum", "count"])
        self._feature_statistics_df["mean"] = self._feature_statistics_df["sum"] / self._feature_statistics_df["count"]
        self._feature_statistics_df.columns = [X_copy.name + "_sum", X_copy.name + "_count", X_copy.name + "_mean"]
        self._feature_statistics_df = self._feature_statistics_df.reset_index()

        # Get all levels that only appear once
        self._single_levels_npa = self._feature_statistics_df.loc[self._feature_statistics_df[X_copy.name + "_count"] == 1].index.values

        return self

    def transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                Data transformed by the transformer.
        """

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # For all levels that appear only once, replace them by the average target value
        X_copy.loc[X_copy.isin(self._single_levels_npa)] = self._target_mean

        # Encode the remaining levels
        X_copy_df = pd.DataFrame(X_copy).merge(self._feature_statistics_df, how = "left", on = X_copy.name)
        X_copy_df.index = X.index

        if y is not None:
            X_copy.loc[~X_copy.isin(self._single_levels_npa)] = (X_copy_df[X_copy.name + "_sum"].loc[~X_copy.isin(self._single_levels_npa)] - y.loc[~X_copy.isin(self._single_levels_npa)]) / (X_copy_df[X_copy.name + "_count"].loc[~X_copy.isin(self._single_levels_npa)] - 1)
        
            if self.add_gaussian_noise:
                X_copy = X_copy * np.random.normal(1., self.sigma, X_copy.shape[0])
        else:
            # Encode the remaining levels
            X_copy.loc[~X_copy.isin(self._single_levels_npa)] = X_copy_df[X_copy.name + "_mean"].loc[~X_copy.isin(self._single_levels_npa)]
                
        return X_copy

    def fit_transform(self, X, y = None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X, y)

class CategoricalFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a transformer that encodes categorical features into numerical ones.
    """
    
    def __init__(self, columns_names_lst, encoders_lst, missing_value_replacement = "NA", drop_initial_features = True):
        """
        Class' constructor

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.

        encoders_lst : list
                Encoders chosen for each column of the columns_names_lst list.

        missing_value_replacement : string
                Value used to replace missing values.

        drop_initial_features : bool
                Flag indicating whether to drop or not initial features used for encoding.
                
        Returns
        -------
        None
        """
        
        if len(columns_names_lst) != len(encoders_lst):
            raise ValueError("Number of items in 'columns_names_lst' doesn't match number of items in 'encoders_lst'!")

        self.columns_names_lst = columns_names_lst
        self.encoders_lst = encoders_lst
        self.missing_value_replacement = missing_value_replacement
        self.drop_initial_features = drop_initial_features
        
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
        self: CategoricalFeaturesEncoder object
                Return current object.
        """

        for idx in range(len(self.columns_names_lst)):
            # Fit each encoder
            if isinstance(self.encoders_lst[idx], LeaveOneOutEncoder) or isinstance(self.encoders_lst[idx], TargetAvgEncoder) or isinstance(self.encoders_lst[idx], GroupingEncoder):
                self.encoders_lst[idx].fit(X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement), y)
            else:
                self.encoders_lst[idx].fit(X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement))
            
        return self

    def transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        start_time = time.time()
        
        for idx in range(len(self.columns_names_lst)):
            # Impute missing values
            X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement, inplace = True)

            # Transform data using each encoder
            if isinstance(self.encoders_lst[idx], LabelBinarizer) or isinstance(self.encoders_lst[idx], GroupingEncoder): # We need a different handling of the preprocessing for one-hot coding based encoders, as they add new columns
                # Get the result of feature transformation and convert it to Pandas DataFrame
                tmp_npa = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]])

                if len(tmp_npa.shape) == 2 and tmp_npa.shape[1] > 1:
                    columns_lst = [self.columns_names_lst[idx] + "_" + level for level in self.encoders_lst[idx].classes_]

                    # Add the DataFrame to the existing data
                    X = pd.concat([X, pd.DataFrame(tmp_npa, index = X.index, columns = columns_lst)], axis = 1)
                else:
                    X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = tmp_npa
                                    
            elif isinstance(self.encoders_lst[idx], LeaveOneOutEncoder):
                X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]], y)
            else:
                X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]])

        # Drop the initial features
        if self.drop_initial_features:
            X.drop(list(set(self.columns_names_lst)), axis = 1, inplace = True)

        print("CategoricalFeaturesEncoder transformed", len(self.columns_names_lst), "categorical features in", round(time.time() - start_time, 3), "seconds.")

        return X

    def fit_transform(self, X, y = None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X, y)
            
###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed for creating a transformer that encodes  #
# text using bag of words, TF-IDF or LSA representation. It is compatible     #
# with the Scikit-Learn framework and uses sparse matrices for reducing memory#
# consumption.                                                                #
#                                                                             #
# Credits for WordBatch: https://github.com/anttttti/Wordbatch                #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-22                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from wordbatch import WordBatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, coo_matrix, hstack
import multiprocessing as mp

class SparseTextEncoder(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a text encoder using bag of words or TF-IDF representation.
    """

    def __init__(self, columns_names_lst, encoders_lst, nnz_threshold = 2, output_format = "csr"):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.

        encoders_lst : list
                Encoders chosen for each column of the columns_names_lst list.

        nnz_threshold: positive integer (default = 2)
                Minimum number of non-zero values we want to have in generated features. If, for a given generated feature, 
                the number of non-zero values is less than this threshold, then the feature is dropped.
                
        output_format: string (default = "csr")
                Output format of this transformer. This can be either "csr" or "pandas". In the first case, the data is 
                returned in a Scipy CSR matrix. In the latter, the data is returned in a Pandas SparseDataFrame.
                The Pandas format keeps columns names, but the conversion to this format takes some time.
                                
        Returns
        -------
        None
        """

        self.columns_names_lst = columns_names_lst
        self.encoders_lst = encoders_lst
        self.nnz_threshold = nnz_threshold
        self.output_format = output_format

        self._encoders_masks_lst = [None for i in encoders_lst] # List of masks that allows to remove columns without enough non-zero values.

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
        self: SparseTextEncoder object
                Return current object.
        """

        raise NotImplementedError("Method not implemented! Please call fit_transform() instead.")

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
        encoded_features_sdf: Pandas SparseDataFrame OR sparse_merge: Scipy CSR matrix
                Transformed data.
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.columns_names_lst)) # Standard columns are columns we don't want to transform
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]
        
        for idx, column in enumerate(self.columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            if type(self.encoders_lst[idx]) == WordBatch:
                self.encoders_lst[idx].dictionary_freeze = True # Freeze dictionary to avoid adding new words when calling transform() method.

            # Encode the feature
            encoded_features_csr = self.encoders_lst[idx].fit_transform(X[column]) 
            
            # Compute mask to remove generated features that don't have enough non-zero values
            if type(self.encoders_lst[idx]) != LSAVectorizer:
                self._encoders_masks_lst[idx] = np.array(np.clip(encoded_features_csr.getnnz(axis = 0) - self.nnz_threshold, 0, 1), dtype = bool)
            
                # Actually remove generated features that don't have enough non-zero values
                encoded_features_csr = encoded_features_csr[:, self._encoders_masks_lst[idx]]
            else:
                encoded_features_csr = coo_matrix(encoded_features_csr)
            
            # Generate the features name
            if type(self.encoders_lst[idx]) == CountVectorizer or type(self.encoders_lst[idx]) == TfidfVectorizer:
                encoded_columns_names_lst = [column + "_" + w for w in self.encoders_lst[idx].get_feature_names()]
            elif type(self.encoders_lst[idx]) == LabelBinarizer:
                encoded_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == WordBatch:
                encoded_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == LSAVectorizer:
                encoded_columns_names_lst = [column + "_LSA_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]

            # If the number of columns names is greater than the number of columns, drop useless column names
            if len(encoded_columns_names_lst) > encoded_features_csr.shape[1]:
                encoded_columns_names_lst = np.array(encoded_columns_names_lst)[self._encoders_masks_lst[idx]].tolist()
                
            arrays_lst.append(encoded_features_csr)
            columns_names_lst.extend(encoded_columns_names_lst)

        sparse_merge = hstack(arrays_lst).tocsr()
        
        if self.output_format == "pandas":
            encoded_features_sdf = pd.SparseDataFrame(sparse_merge, default_fill_value = 0, columns = columns_names_lst, index = X.index)
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", encoded_features_sdf.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return encoded_features_sdf
        else:
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", sparse_merge.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
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
        encoded_features_sdf: Pandas SparseDataFrame OR sparse_merge: Scipy CSR matrix
                Transformed data.
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]
        
        for idx, column in enumerate(self.columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            # Encode the feature
            encoded_features_csr = self.encoders_lst[idx].transform(X[column])

            if type(self.encoders_lst[idx]) != LSAVectorizer:
                # Remove generated features that don't have enough non-zero values
                encoded_features_csr = encoded_features_csr[:, self._encoders_masks_lst[idx]]
            else:
                encoded_features_csr = coo_matrix(encoded_features_csr)

            # Generate the features name
            if type(self.encoders_lst[idx]) == CountVectorizer or type(self.encoders_lst[idx]) == TfidfVectorizer:
                encoded_columns_names_lst = [column + "_" + w for w in self.encoders_lst[idx].get_feature_names()]
            elif type(self.encoders_lst[idx]) == LabelBinarizer:
                encoded_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == WordBatch:
                encoded_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == LSAVectorizer:
                encoded_columns_names_lst = [column + "_LSA_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]

            # If the number of columns names is greater than the number of columns, drop useless column names
            if len(encoded_columns_names_lst) > encoded_features_csr.shape[1]:
                encoded_columns_names_lst = np.array(encoded_columns_names_lst)[self._encoders_masks_lst[idx]].tolist()

            arrays_lst.append(encoded_features_csr)
            columns_names_lst.extend(encoded_columns_names_lst)
            
        sparse_merge = hstack(arrays_lst).tocsr()
        
        if self.output_format == "pandas":
            encoded_features_sdf = pd.SparseDataFrame(sparse_merge, default_fill_value = 0, columns = columns_names_lst, index = X.index)
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", encoded_features_sdf.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return encoded_features_sdf
        else:
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", sparse_merge.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return sparse_merge

class LSAVectorizer(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a Latent Semantic Analysis.
    """

    def __init__(self, lsa_components, tfidf_parameters = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10}):
        """
        This is the class' constructor.

        Parameters
        ----------
        lsa_components : positive integer
                Number of components we want to keep.

        tfidf_parameters : dict (default = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10})
                Dict containing parameters of TfidfVectorizer used in this class.
                Each dictionary key must corresponds to one scikit-learn TfidfVectorizer parameter,
                as defined here: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        Returns
        -------
        None
        """
        
        self.lsa_components = lsa_components
        self.tfidf_parameters = tfidf_parameters

        self._tfv = TfidfVectorizer(**self.tfidf_parameters)
        self._svd = TruncatedSVD(self.lsa_components)
        
    def fit(self, X, y = None):
        """
        Fit the transformer on provided text.

        Parameters
        ----------
        X : pd.Series
                Series containing the text to transform.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        self: LSAVectorizer object
                Return current object.
        """

        # Remove missing values
        X.fillna("NA", inplace = True)

        # Fit the TfidfVectorizer
        tfidf_output_csr = self._tfv.fit_transform(X, y)

        # Fit the SVD
        self._svd.fit(tfidf_output_csr)

        print("LSA explained variance:", round(np.sum(self._svd.explained_variance_ratio_) * 100, 3), "%")

        return self
    
    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                Data that will be transformed.
                
        Returns
        -------
        lsa_output_npa : numpy array
                Transformed data.
        """

        # Remove missing values
        X.fillna("NA", inplace = True)

        # Transform the data using TfidfVectorizer
        tfidf_output_csr = self._tfv.transform(X)

        # Reduce dimensionality using SVD
        lsa_output_npa = self._svd.transform(tfidf_output_csr)
                
        return lsa_output_npa

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains code used to generate classification predictions for     #
# Quadratic Weighted Kappa metric from predictions generated using regression #
# model.                                                                      #
# Code inspired from:                                                         #
# https://www.kaggle.com/fiancheto/petfinder-simple-lgbm-baseline-lb-0-399    #
#                                                                             #
###############################################################################

import numpy as np
import scipy as sp

from functools import partial
from ml_metrics import quadratic_weighted_kappa

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        # X = pred_test_y, y = y_eval

        #loss_partial = partial(self._kappa_loss, X=X, y=y)
        #initial_coef = [0.5, 1.5, 2.5, 3.5]
        #self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = "nelder-mead")["x"]
        self.coef_ = [0.0, 0.0, 0.0, 0.0]
        train_distribution_df = pd.Series(y).value_counts().reset_index()
        train_distribution_df.sort_values("index", ascending = True, inplace = True)
        train_distribution_dict = list(train_distribution_df.set_index("index").cumsum().to_dict().values())[0]

        X2 = pd.Series(X).sort_values(ascending = True).values
        for i in range(4):
            self.coef_[i] = round(X2[train_distribution_dict[i]], 3)

    def predict(self, X):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < self.coef_[0]:
                X_p[i] = 0
            elif pred >= self.coef_[0] and pred < self.coef_[1]:
                X_p[i] = 1
            elif pred >= self.coef_[1] and pred < self.coef_[2]:
                X_p[i] = 2
            elif pred >= self.coef_[2] and pred < self.coef_[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file is an advanced wrapper for the package LightGBM that is           #
# compatible with scikit-learn API.                                           #
#                                                                             #
# Credits for LightGBM: https://github.com/Microsoft/LightGBM                 #
#                                                                             #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-01-02                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, cohen_kappa_score

from sklearn.model_selection import StratifiedKFold
from ml_metrics import quadratic_weighted_kappa
from collections import Counter

class BlendedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    The purpose of this class is to provide a wrapper for a blended LightGBM model.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, maximize = True, nrounds = 10000, random_state = 0, eval_size = 0.1, eval_split_type = "random", verbose_eval = 1):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.

        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as loss function.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        eval_size : float between 0 and 1.
                This indicates the size of the test set. Not used when enable_cv is True.

        eval_split_type : string, either "random" or "time"
                This indicates the type of split for evaluation: random for iid samples and time for time series. Not used when enable_cv is True.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.custom_eval_function = custom_eval_function
        self.maximize = maximize
        self.nrounds = nrounds
        self.random_state = random_state
        self.eval_size = eval_size
        self.eval_split_type = eval_split_type
        self.verbose_eval = verbose_eval

        self.lgb_model_lst = []
        self._optimized_rounder_lst = []
        self.model_name = "BlendedLightGBM"

        self.num_class = self.params.pop("num_class")
        self.params["metric"] = "rmse"
        self.params["application"] = "regression"
        
    def fit(self, X, y):
        """
        This method trains the BlendedLightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.

        stratified_split : bool (default = False)
                This flag indicates whether to make a stratified split or not.

        return_eval : bool (default = False)
                This flag indicates whether to return eval data or not.
                
        Returns
        -------
        None
        """
        
        print("LightGBM training...")
        label='lgb'
        kf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
        fold_splits = kf.split(X, y)
        qwk_scores = []
        pred_full_test = 0
        pred_train = np.zeros((X.shape[0], 5))
        self._all_coefficients = np.zeros((5, 4))
        self._feature_importance_df = pd.DataFrame()
        i = 1

        for train_index, eval_index in fold_splits:
            print('Started ' + label + ' fold ' + str(i) + '/5')
            X_train = X.values[train_index]
            X_eval = X.values[eval_index]
            y_train = y.values[train_index]
            y_eval = y.values[eval_index]

            dtrain = lgb.Dataset(X_train, label = y_train)
            dvalid = lgb.Dataset(X_eval, label = y_eval)
            watchlist = [dtrain, dvalid]

            print('Train LGB')
            lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval)

            print('Predict 1/2')
            pred_test_y = lgb_model.predict(X_eval, num_iteration = lgb_model.best_iteration)
            optR = OptimizedRounder()
            optR.fit(pred_test_y, y_eval)
            coefficients = optR.coefficients()
            pred_test_y_k = optR.predict(pred_test_y)
            print("Valid Counts = ", Counter(y_eval))
            print("Predicted Counts = ", Counter(pred_test_y_k))
            print("Coefficients = ", coefficients)
            qwk = quadratic_weighted_kappa(y_eval, pred_test_y_k)
            print("QWK = ", qwk)

            print('Predict 2/2')
            pred_val_y = pred_test_y.reshape(-1, 1)
            
            importances = lgb_model.feature_importance(importance_type = "gain")

            pred_train[eval_index] = pred_val_y
            self._all_coefficients[i-1, :] = coefficients
            qwk_scores.append(qwk)
            print(label + ' cv score {}: QWK {}'.format(i, qwk))
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = X.columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            self._feature_importance_df = pd.concat([self._feature_importance_df, fold_importance_df], axis=0)        
            i += 1

            # Save LGBM model
            self.lgb_model_lst.append(lgb_model)
            self._optimized_rounder_lst.append(optR)

        print('{} cv QWK scores : {}'.format(label, qwk_scores))
        print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
        print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
        
        print("LightGBM training... done")

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

        pred_full_test = 0

        for i in range(5):
            pred_test_y2 = self.lgb_model_lst[i].predict(X, num_iteration = self.lgb_model_lst[i].best_iteration)
            pred_test_y = pred_test_y2.reshape(-1, 1)
            pred_full_test = pred_full_test + pred_test_y

        pred_full_test = pred_full_test / 5.0

        coefficients_ = np.mean(self._all_coefficients, axis = 0)
        optR = OptimizedRounder()
        optR.coef_ = coefficients_
        predictions_npa = optR.predict(pred_full_test.flatten()).astype(np.int32)

        return predictions_npa

    def get_features_importance(self, importance_type = "gain"):
        """
        This method gets model's features importance.

        Parameters
        ----------
        importance_type : string, optional (default = "gain")
                How the importance is calculated. If "split", result contains numbers of times the feature is used in a model. 
                If "gain", result contains total gains of splits which use the feature.
                                
        Returns
        -------
        self._feature_importance_df : Pandas Data Frame
                Feature importance of each feature of the dataset.
        """
        
        self._feature_importance_df = self._feature_importance_df.groupby("feature")["feature", "importance"].mean().reset_index()
        self._feature_importance_df = self._feature_importance_df.sort_values(by = "importance", ascending = False).reset_index(drop = True)

        return self._feature_importance_df

#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains the code needed for the preprocessing step of metadata   #
# JSON files.                                                                 #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-02-17                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import json
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp


class MetadataFilesPreprocessingStep(object):
    """
    This class extracts the metadata contained in JSON files.
    Some parts of the code of this class comes from https://www.kaggle.com/wrosinski/baselinemodeling.
    """

    def __init__(self, n_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        n_cores : integer (default = -1)
                Number of CPU cores to use to extract the data. 
                If n_cores == 1, then multiprocessing module is not used.
                If n_cores == -1, then all CPU cores are used.

        Returns
        -------
        None
        """

        self.n_cores = n_cores

        if n_cores == -1:
            self.n_cores = mp.cpu_count()

        self.sentence_sep = " "

    def _parse_metadata_file(self, filename):
        """
        This method actually extracts data from metadata JSON files.

        Parameters
        ----------
        filename : string
                Path of the file to extract.
                                
        Returns
        -------
        file_metadata_df : pd.DataFrame
                Data frame containing the extracted data.
        """
        
        with open(filename, "r", encoding = "UTF-8") as f:
            metadata_file_dict = json.load(f)

        file_keys = list(metadata_file_dict.keys())
        
        # Extract data from "labelAnnotations" key
        if "labelAnnotations" in file_keys:
            file_annotations_lst = metadata_file_dict["labelAnnotations"]#[:int(len(metadata_file_dict["labelAnnotations"]) * 0.3)] # WARNING: Hard-coded threshold (0.3) !!!
            file_annotations_scores_npa = np.asarray([x["score"] for x in file_annotations_lst])
            file_annots_max_score = file_annotations_scores_npa.max()
            file_annots_min_score = file_annotations_scores_npa.min()
            file_annots_avg_score = file_annotations_scores_npa.mean()
            file_annots_std_score = file_annotations_scores_npa.std()
            file_top_desc = [x["description"].replace(" ", "_") for x in file_annotations_lst]
        else:
            file_annots_max_score = np.nan
            file_annots_min_score = np.nan
            file_annots_avg_score = np.nan
            file_annots_std_score = np.nan
            file_top_desc = [""]
        
        # Extract data from "imagePropertiesAnnotation/dominantColors/colors"
        file_colors_lst = metadata_file_dict["imagePropertiesAnnotation"]["dominantColors"]["colors"]
        file_color_score_npa = np.asarray([x["score"] for x in file_colors_lst])
        file_color_max_score = file_color_score_npa.max()
        file_color_min_score = file_color_score_npa.min()
        file_color_avg_score = file_color_score_npa.mean()
        file_color_std_score = file_color_score_npa.std()
        file_color_pixelfrac_score_npa = np.asarray([x["pixelFraction"] for x in file_colors_lst])
        file_color_pixelfrac_max_score = file_color_score_npa.max()
        file_color_pixelfrac_min_score = file_color_score_npa.min()
        file_color_pixelfrac_avg_score = file_color_score_npa.mean()
        file_color_pixelfrac_std_score = file_color_score_npa.std()
        file_color_red_value_npa = np.asarray([x["color"]["red"] if "red" in x["color"] else 0 for x in file_colors_lst])
        file_color_red_max_value = file_color_red_value_npa.max()
        file_color_red_min_value = file_color_red_value_npa.min()
        file_color_red_avg_value = file_color_red_value_npa.mean()
        file_color_red_std_value = file_color_red_value_npa.std()
        file_color_green_value_npa = np.asarray([x["color"]["green"] if "green" in x["color"] else 0 for x in file_colors_lst])
        file_color_green_max_value = file_color_green_value_npa.max()
        file_color_green_min_value = file_color_green_value_npa.min()
        file_color_green_avg_value = file_color_green_value_npa.mean()
        file_color_green_std_value = file_color_green_value_npa.std()
        file_color_blue_value_npa = np.asarray([x["color"]["blue"] if "blue" in x["color"] else 0 for x in file_colors_lst])
        file_color_blue_max_value = file_color_blue_value_npa.max()
        file_color_blue_min_value = file_color_blue_value_npa.min()
        file_color_blue_avg_value = file_color_blue_value_npa.mean()
        file_color_blue_std_value = file_color_blue_value_npa.std()

        # Extract data from "cropHintsAnnotation/cropHints"
        file_crops_lst = metadata_file_dict["cropHintsAnnotation"]["cropHints"]
        file_crop_conf_npa = np.asarray([x["confidence"] for x in file_crops_lst])
        file_crop_conf_max = file_crop_conf_npa.max()
        file_crop_conf_min = file_crop_conf_npa.min()
        file_crop_conf_avg = file_crop_conf_npa.mean()
        
        if "importanceFraction" in file_crops_lst[0].keys():
            file_crop_importance_npa = np.asarray([x["importanceFraction"] for x in file_crops_lst])
            file_crop_importance_max = file_crop_importance_npa.max()
            file_crop_importance_min = file_crop_importance_npa.min()
            file_crop_importance_avg = file_crop_importance_npa.mean()
        else:
            file_crop_importance_max = np.nan
            file_crop_importance_min = np.nan
            file_crop_importance_avg = np.nan

        file_metadata_dict = {
            "annots_max_score": file_annots_max_score,
            "annots_min_score": file_annots_min_score,
            "annots_avg_score": file_annots_avg_score,
            "annots_std_score": file_annots_std_score,
            "color_max_score": file_color_max_score,
            "color_min_score": file_color_min_score,
            "color_avg_score": file_color_avg_score,
            "color_std_score": file_color_std_score,
            "color_max_pixelfrac": file_color_pixelfrac_max_score,
            "color_min_pixelfrac": file_color_pixelfrac_min_score,
            "color_avg_pixelfrac": file_color_pixelfrac_avg_score,
            "color_std_pixelfrac": file_color_pixelfrac_std_score,
            "color_red_max_value": file_color_red_max_value,
            "color_red_min_value": file_color_red_min_value,
            "color_red_avg_value": file_color_red_avg_value,
            "color_red_std_value": file_color_red_std_value,
            "color_green_max_value": file_color_green_max_value,
            "color_green_min_value": file_color_green_min_value,
            "color_green_avg_value": file_color_green_avg_value,
            "color_green_std_value": file_color_green_std_value,
            "color_blue_max_value": file_color_blue_max_value,
            "color_blue_min_value": file_color_blue_min_value,
            "color_blue_avg_value": file_color_blue_avg_value,
            "color_blue_std_value": file_color_blue_std_value,
            "crop_max_conf": file_crop_conf_max,
            "crop_min_conf": file_crop_conf_min,
            "crop_avg_conf": file_crop_conf_avg,
            "crop_max_importance": file_crop_importance_max,
            "crop_min_importance": file_crop_importance_min,
            "crop_avg_importance": file_crop_importance_avg,
            "annots_top_desc": self.sentence_sep.join(file_top_desc)
        }
        
        file_metadata_df = pd.DataFrame.from_dict(file_metadata_dict, orient = "index").T
        file_metadata_df = file_metadata_df.add_prefix("metadata_")
        
        return file_metadata_df

    def _extract_data_from_metadata_files_helper(self, pet_id, json_files_dir_path_str):
        """
        This method is a helper function used to extract data from metadata JSON files.

        Parameters
        ----------
        pet_id : string
                ID of the pet we want to extract metadata for.

        json_files_dir_path_str : string
                Path of the directory containing the JSON files.
                
        Returns
        -------
        metadata_df : pd.DataFrame
                Data frame containing the extracted data. This is None if no metadata were found.
        """
                                    
        metadata_filenames_lst = sorted(glob.glob(json_files_dir_path_str + "{}*.json".format(pet_id)))

        if len(metadata_filenames_lst) > 0:
            metadata_df_lst = []
            for f in metadata_filenames_lst:
                current_metadata_df = self._parse_metadata_file(f)
                current_metadata_df["PetID"] = pet_id
                metadata_df_lst.append(current_metadata_df)

            metadata_df = pd.concat(metadata_df_lst, ignore_index = True, sort = False)
        else:
            metadata_df = None

        return metadata_df

    def extract_data_from_metadata_files(self, pets_id_lst, json_files_dir_path_str):
        """
        This method is used to extract data from metadata JSON files.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        json_files_dir_path_str : string
                Path of the directory containing the JSON files.
                
        Returns
        -------
        metadata_desc_df, grouped_metadata_df : pd.DataFrame
                Data frame containing the extracted data.
        """
                            
        st = time.time()
        print("Preprocessing metadata files found in ", json_files_dir_path_str, "...") 
        
        extracted_metadata_lst = Parallel(n_jobs = self.n_cores, verbose = 1)(delayed(self._extract_data_from_metadata_files_helper)(i, json_files_dir_path_str = json_files_dir_path_str) for i in pets_id_lst)
        extracted_metadata_lst = [x for x in extracted_metadata_lst if x is not None]

        extracted_metadata_df = pd.concat(extracted_metadata_lst, ignore_index = True, sort = False)

        print("    Grouping extracted features by 'PetID'...")
        
        metadata_desc_df = extracted_metadata_df.groupby(["PetID"])["metadata_annots_top_desc"].unique()
        metadata_desc_df = metadata_desc_df.reset_index()
        metadata_desc_df["metadata_annots_top_desc"] = metadata_desc_df["metadata_annots_top_desc"].apply(lambda x: self.sentence_sep.join(x))

        grouped_metadata_df = extracted_metadata_df.drop(["metadata_annots_top_desc"], axis = 1)
        for i in grouped_metadata_df.columns:
            if "PetID" not in i:
                grouped_metadata_df[i] = grouped_metadata_df[i].astype(float)

        grouped_metadata_df = grouped_metadata_df.groupby(["PetID"]).agg(["mean", "sum"])
        grouped_metadata_df.columns = pd.Index(["{}_{}_{}".format("metadata", c[0], c[1]) for c in grouped_metadata_df.columns.tolist()])
        grouped_metadata_df = grouped_metadata_df.reset_index()

        print("Preprocessing metadata files... done in", round(time.time() - st, 3), "secs")

        return metadata_desc_df, grouped_metadata_df

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains the code needed for the preprocessing step of sentiment  #
# JSON files.                                                                 #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-02-17                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import json
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp

class SentimentFilesPreprocessingStep(object):
    """
    This class extracts the sentiment data contained in JSON files.
    Some parts of the code of this class comes from https://www.kaggle.com/wrosinski/baselinemodeling.
    """

    def __init__(self, n_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        n_cores : integer (default = -1)
                Number of CPU cores to use to extract the data. 
                If n_cores == 1, then multiprocessing module is not used.
                If n_cores == -1, then all CPU cores are used.

        Returns
        -------
        None
        """

        self.n_cores = n_cores

        if n_cores == -1:
            self.n_cores = mp.cpu_count()

        self.sentence_sep = " "
        
    def _parse_sentiment_file(self, filename):
        """
        This method actually extracts data from sentiment JSON files.

        Parameters
        ----------
        filename : string
                Path of the file to extract.
                                
        Returns
        -------
        file_sentiments_df : pd.DataFrame
                Data frame containing the extracted data.
        """
        
        with open(filename, "r", encoding = "UTF-8") as f:
            sentiment_file_dict = json.load(f)

        # Extract the language from the text
        file_language = sentiment_file_dict["language"]
        
        # Extract whole document sentiment
        ## score: overall emotional leaning of the text
        ## magnitude: overall strength of emotion (both positive and negative) within the given text
        file_overall_score = sentiment_file_dict["documentSentiment"]["score"]
        file_overall_magnitude = sentiment_file_dict["documentSentiment"]["magnitude"]

        # Look for entities
        file_entities_count = len(sentiment_file_dict["entities"])
        if file_entities_count > 0:
            file_entities_name_lst = [x["name"] for x in sentiment_file_dict["entities"]]
            file_entities_type_lst = [x["type"] for x in sentiment_file_dict["entities"]]
            file_entities_salience_npa = np.asarray([x["salience"] for x in sentiment_file_dict["entities"]])
            file_entities_max_salience = file_entities_salience_npa.max()
            file_entities_min_salience = file_entities_salience_npa.min()
            file_entities_avg_salience = file_entities_salience_npa.mean()
            file_entities_std_salience = file_entities_salience_npa.std()
        else:
            file_entities_name_lst = []
            file_entities_type_lst = []
            file_entities_max_salience = np.nan
            file_entities_min_salience = np.nan
            file_entities_avg_salience = np.nan
            file_entities_std_salience = np.nan
        
        # Look for sentences sentiments
        file_sentences_count = len(sentiment_file_dict["sentences"])
        if file_sentences_count > 0:
            file_sentences_sentiment_score_npa = np.asarray([x["sentiment"]["score"] for x in sentiment_file_dict["sentences"]])
            file_sentences_sentiment_magnitude_npa = np.asarray([x["sentiment"]["magnitude"] for x in sentiment_file_dict["sentences"]])
            file_sentences_max_sentiment_score = file_sentences_sentiment_score_npa.max()
            file_sentences_min_sentiment_score = file_sentences_sentiment_score_npa.min()
            file_sentences_avg_sentiment_score = file_sentences_sentiment_score_npa.mean()
            file_sentences_std_sentiment_score = file_sentences_sentiment_score_npa.std()
            file_sentences_max_sentiment_magnitude = file_sentences_sentiment_magnitude_npa.max()
            file_sentences_min_sentiment_magnitude = file_sentences_sentiment_magnitude_npa.min()
            file_sentences_avg_sentiment_magnitude = file_sentences_sentiment_magnitude_npa.mean()
            file_sentences_std_sentiment_magnitude = file_sentences_sentiment_magnitude_npa.std()
        else:
            file_sentences_max_sentiment_score = np.nan
            file_sentences_min_sentiment_score = np.nan
            file_sentences_avg_sentiment_score = np.nan
            file_sentences_std_sentiment_score = np.nan
            file_sentences_max_sentiment_magnitude = np.nan
            file_sentences_min_sentiment_magnitude = np.nan
            file_sentences_avg_sentiment_magnitude = np.nan
            file_sentences_std_sentiment_magnitude = np.nan

        file_sentiments_dict = {
            "file_language": file_language,
            "file_overall_score": file_overall_score,
            "file_overall_magnitude": file_overall_magnitude,
            "file_entities_name_lst": self.sentence_sep.join(file_entities_name_lst),
            "file_entities_type_lst": self.sentence_sep.join(file_entities_type_lst),
            "file_entities_max_salience": file_entities_max_salience,
            "file_entities_min_salience": file_entities_min_salience,
            "file_entities_avg_salience": file_entities_avg_salience,
            "file_entities_std_salience": file_entities_std_salience,
            "file_entities_count": file_entities_count,
            "file_sentences_max_sentiment_score": file_sentences_max_sentiment_score,
            "file_sentences_min_sentiment_score": file_sentences_min_sentiment_score,
            "file_sentences_avg_sentiment_score": file_sentences_avg_sentiment_score,
            "file_sentences_std_sentiment_score": file_sentences_std_sentiment_score,
            "file_sentences_max_sentiment_magnitude": file_sentences_max_sentiment_magnitude,
            "file_sentences_min_sentiment_magnitude": file_sentences_min_sentiment_magnitude,
            "file_sentences_avg_sentiment_magnitude": file_sentences_avg_sentiment_magnitude,
            "file_sentences_std_sentiment_magnitude": file_sentences_std_sentiment_magnitude,
            "file_sentences_count": file_sentences_count
        }

        file_sentiments_df = pd.DataFrame.from_dict(file_sentiments_dict, orient = "index").T
        file_sentiments_df = file_sentiments_df.add_prefix("sentiment_")
                
        return file_sentiments_df

    def _extract_data_from_sentiment_files_helper(self, pet_id, json_files_dir_path_str):
        """
        This method is a helper function used to extract data from sentiment JSON files.

        Parameters
        ----------
        pet_id : string
                ID of the pet we want to extract metadata for.

        json_files_dir_path_str : string
                Path of the directory containing the JSON files.
                
        Returns
        -------
        sentiment_df : pd.DataFrame
                Data frame containing the extracted data. This is None if no sentiment data were found.
        """

        sentiment_filename_str = json_files_dir_path_str + "{}.json".format(pet_id)
        try:
            sentiment_df = self._parse_sentiment_file(sentiment_filename_str)
            sentiment_df["PetID"] = pet_id

        except FileNotFoundError:
            sentiment_df = None

        return sentiment_df

    def extract_data_from_sentiment_files(self, pets_id_lst, json_files_dir_path_str):
        """
        This method is used to extract data from sentiment JSON files.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        json_files_dir_path_str : string
                Path of the directory containing the JSON files.
                
        Returns
        -------
        sentiment_desc_df, grouped_sentiment_df : pd.DataFrame
                Data frame containing the extracted data.
        """
                            
        st = time.time()
        print("Preprocessing sentiment files found in ", json_files_dir_path_str, "...")   
        
        extracted_sentiments_lst = Parallel(n_jobs = self.n_cores, verbose = 1)(delayed(self._extract_data_from_sentiment_files_helper)(i, json_files_dir_path_str = json_files_dir_path_str) for i in pets_id_lst)
        extracted_sentiments_lst = [x for x in extracted_sentiments_lst if x is not None]

        extracted_sentiments_df = pd.concat(extracted_sentiments_lst, ignore_index = True, sort = False)

        print("    Grouping extracted features by 'PetID'...")

        sentiment_name_df = extracted_sentiments_df.groupby(["PetID"])["sentiment_file_entities_name_lst"].unique()
        sentiment_name_df = sentiment_name_df.reset_index()
        sentiment_name_df["sentiment_file_entities_name_lst"] = sentiment_name_df["sentiment_file_entities_name_lst"].apply(lambda x: " ".join(x))

        sentiment_type_df = extracted_sentiments_df.groupby(["PetID"])["sentiment_file_entities_type_lst"].unique()
        sentiment_type_df = sentiment_type_df.reset_index()
        sentiment_type_df["sentiment_file_entities_type_lst"] = sentiment_type_df["sentiment_file_entities_type_lst"].apply(lambda x: " ".join(x))

        sentiment_desc_df = sentiment_name_df.merge(sentiment_type_df, how = "left", on = "PetID")

        grouped_sentiment_df = extracted_sentiments_df.drop(["sentiment_file_entities_name_lst", "sentiment_file_entities_type_lst"], axis = 1)
        for i in grouped_sentiment_df.columns:
            if i not in ["PetID", "sentiment_file_language"]:
                grouped_sentiment_df[i] = grouped_sentiment_df[i].astype(float)

        language_df = grouped_sentiment_df[["PetID", "sentiment_file_language"]]
        grouped_sentiment_df.drop("sentiment_file_language", axis = 1, inplace = True)
        grouped_sentiment_df = grouped_sentiment_df.groupby(["PetID"]).agg(["mean", "sum"])
        grouped_sentiment_df.columns = pd.Index(["{}_{}_{}".format("sentiment", c[0], c[1]) for c in grouped_sentiment_df.columns.tolist()])
        grouped_sentiment_df = grouped_sentiment_df.reset_index()
        grouped_sentiment_df = grouped_sentiment_df.merge(language_df, how = "left", on = "PetID")

        print("Preprocessing sentiment files... done in", round(time.time() - st, 3), "secs")

        return sentiment_desc_df, grouped_sentiment_df
        
###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains the code needed to extract metadata from images of each  #
# pet.                                                                        #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-03-09                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import json
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import cv2
import os

import numpy as np
import pandas as pd
import time
import json
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import cv2
import os

class ImagesMetadataExtractionStep(object):
    """
    This class extracts the metadata from each image provided with pets data.
    """

    def __init__(self, n_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        n_cores : integer (default = -1)
                Number of CPU cores to use to extract the data. 
                If n_cores == 1, then multiprocessing module is not used.
                If n_cores == -1, then all CPU cores are used.

        Returns
        -------
        None
        """

        self.n_cores = n_cores

        if n_cores == -1:
            self.n_cores = mp.cpu_count()

    def _extract_metadata_from_image(self, filename):
        """
        This method actually extracts the metadata from the image.

        Parameters
        ----------
        filename : string
                Path of the file to extract.
                                
        Returns
        -------
        file_metadata_df : pd.DataFrame
                Data frame containing the extracted data.
        """
        
        img = cv2.imread(filename)

        # Look at image quality: blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_amt = cv2.Laplacian(gray, cv2.CV_64F).std() 

        # Compute HU moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
       
        # Log scale HU moments
        hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        file_metadata_dict = {
            "file_size": os.path.getsize(filename),
            "height": img.shape[0],
            "width": img.shape[1],
            "nb_pixels": img.shape[0] * img.shape[1],
            "blur_amt": blur_amt,
            "hu_moment_0": hu_moments[0],
            "hu_moment_1": hu_moments[1],
            "hu_moment_2": hu_moments[2],
            "hu_moment_3": hu_moments[3],
            "hu_moment_4": hu_moments[4],
            "hu_moment_5": hu_moments[5],
            "hu_moment_6": hu_moments[6]
        }
                
        file_metadata_df = pd.DataFrame.from_dict(file_metadata_dict, orient = "index").T
        file_metadata_df = file_metadata_df.add_prefix("image_metadata_")
        
        return file_metadata_df

    def _extract_data_from_images_helper(self, pet_id, images_dir_path_str):
        """
        This method is a helper function used to extract metadata from pets' images.

        Parameters
        ----------
        pet_id : string
                ID of the pet we want to extract metadata for.

        images_dir_path_str : string
                Path of the directory containing the images.
                
        Returns
        -------
        metadata_df : pd.DataFrame
                Data frame containing the extracted data. This is None if no metadata were found.
        """
                                    
        metadata_filenames_lst = sorted(glob.glob(images_dir_path_str + "{}*.*".format(pet_id)))

        if len(metadata_filenames_lst) > 0:
            metadata_df_lst = []
            for f in metadata_filenames_lst:
                current_metadata_df = self._extract_metadata_from_image(f)
                current_metadata_df["PetID"] = pet_id
                metadata_df_lst.append(current_metadata_df)

            metadata_df = pd.concat(metadata_df_lst, ignore_index = True, sort = False)
        else:
            metadata_df = None

        return metadata_df

    def extract_metadata_from_images(self, pets_id_lst, images_dir_path_str):
        """
        This method is used to extract metadata from pets' images.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        images_dir_path_str : string
                Path of the directory containing the images.
                
        Returns
        -------
        metadata_desc_df, grouped_metadata_df : pd.DataFrame
                Data frame containing the extracted data.
        """
                            
        st = time.time()
        print("Extracting metadata from images found in ", images_dir_path_str, "...") 
        
        extracted_metadata_lst = Parallel(n_jobs = self.n_cores, verbose = 1)(delayed(self._extract_data_from_images_helper)(i, images_dir_path_str = images_dir_path_str) for i in pets_id_lst)
        extracted_metadata_lst = [x for x in extracted_metadata_lst if x is not None]

        extracted_metadata_df = pd.concat(extracted_metadata_lst, ignore_index = True, sort = False)

        print("    Grouping extracted features by 'PetID'...")
        for i in extracted_metadata_df.columns:
            if "PetID" not in i:
                extracted_metadata_df[i] = extracted_metadata_df[i].astype(float)

        extracted_metadata_df = extracted_metadata_df.groupby(["PetID"]).agg(["mean", "std"])
        extracted_metadata_df.columns = pd.Index(["{}_{}_{}".format("img_metadata", c[0], c[1]) for c in extracted_metadata_df.columns.tolist()])
        extracted_metadata_df = extracted_metadata_df.reset_index()

        print("Extracting metadata from images... done in", round(time.time() - st, 3), "secs")

        return extracted_metadata_df
 
###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains the code needed to extract features from images of each  #
# pet.                                                                        #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-03-09                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import json
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import cv2
import os
import mxnet as mx

class GenerateMxNetRecordIOFile(object):
    """
    This class generates a binary file following RecordIO format from a list of images.
    Some code of this class is copied from https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py.
    """

    def __init__(self, n_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        n_cores : integer (default = -1)
                Number of CPU cores to use to extract the data. 
                If n_cores == 1, then multiprocessing module is not used.
                If n_cores == -1, then all CPU cores are used.

        Returns
        -------
        None
        """

        self.n_cores = n_cores

        if n_cores == -1:
            self.n_cores = mp.cpu_count()

    def _generate_lst_file(self, pets_id_lst, input_dir_path_str, output_file_path_str):
        """
        This method generates the RecordIO .lst file.
        File format is described here: https://mxnet.incubator.apache.org/versions/master/faq/recordio.html?highlight=rec%20file.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        input_dir_path_str: string
                Path of the directory containing images to use.

        output_file_path_str: string
                Path where the .lst file will be written.

        Returns
        -------
        lst_file_content_lst: list of lists
                List of lists in the following format: [integer_image_index, path_to_image, label_index]
        """

        # Get images list
        images_lst = [os.path.basename(f) for f in glob.glob(input_dir_path_str + "*.*") if os.path.isfile(f)]

        # Create the file content
        lst_file_content_df = pd.DataFrame({"integer_image_index": list(range(len(images_lst))), "label_index": [0.000000 for _ in range(len(images_lst))], "path_to_image": images_lst})
        lst_file_content_df["PetID"] = lst_file_content_df["path_to_image"].apply(lambda x: x.split("-")[0])
        lst_file_content_df = lst_file_content_df.loc[lst_file_content_df["PetID"].isin(pets_id_lst)]
        lst_file_content_df.drop("PetID", axis = 1, inplace = True)

        print("    Found", lst_file_content_df.shape[0], "images.")

        # Ensure the DataFrame has the correct column order
        lst_file_content_df = lst_file_content_df[["integer_image_index", "label_index", "path_to_image"]]

        # Save the .lst file
        lst_file_content_df.to_csv(output_file_path_str, sep = "\t", header = False, index = False)

        # Reshape the DataFrame
        lst_file_content_df = lst_file_content_df[["integer_image_index", "path_to_image", "label_index"]]

        # Convert the DataFrame to list
        lst_file_content_lst = lst_file_content_df.values.tolist()

        return lst_file_content_lst

    def _squarify_image(self, img_path, output_size):
        """
        This method reshapes an image to a square format by adding black borders
        on the shortest dimension. Then it resizes the image to 'output_size'.

        Code inspired from 'resize_to_square' function from https://www.kaggle.com/ranjoranjan/single-xgboost-model/notebook.

        Parameters
        ----------
        img_path: string
                Path of the image to transform.

        output_size: positive integer
                Size of the output image.

        Returns
        -------
        None
        """

        im = cv2.imread(img_path)
        old_size = im.shape[:2]
        ratio = float(output_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = output_size - new_size[1]
        delta_h = output_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)

        return new_im

    def _read_worker(self, input_dir_path_str, q_in, q_out):
        """
        This method gets an image, preprocess it and put in the output queue.

        Parameters
        ----------
        input_dir_path_str: string
                Path of the directory containing images to use.

        q_in: Multiprocessing Queue
                Input queue containing images names.

        q_out: Multiprocessing Queue
                Output queue containing result of processing.

        Returns
        -------
        None
        """

        while True:
            deq = q_in.get()
            
            if deq is None:
                break

            i, item = deq

            # Compute the image full path
            img_path_str = os.path.join(input_dir_path_str, item[1])

            # Read and process the image
            img = self._squarify_image(img_path_str, 224)
                        
            # Create one RecordIO item
            header = mx.recordio.IRHeader(0, item[2], item[0], 0)

            if img is None:
                print("Image was None for file: %s" % img_path_str)
                q_out.put((i, None, item))
            else:
                try:
                    s = mx.recordio.pack_img(header, img, quality = 95, img_fmt = ".jpg")
                    q_out.put((i, s, item))
                except Exception as e:
                    print("pack_img error on file: %s" % img_path_str, e)
                    q_out.put((i, None, item))

    def _write_worker(self, q_out, output_file_prefix_str):
        """
        This method fetches a processed image from the output queue and write it to the .rec file.

        Parameters
        ----------
        q_out: Multiprocessing Queue
                Output queue containing result of processing.

        output_file_prefix_str: string
                Prefix indicating where both .rec and .idx files will be written.

        Returns
        -------
        None
        """

        pre_time = time.time()
        count = 0
        record = mx.recordio.MXIndexedRecordIO(output_file_prefix_str + ".idx", output_file_prefix_str + ".rec", "w")
        buf = {}
        more = True
        while more:
            deq = q_out.get()
            if deq is not None:
                i, s, item = deq
                buf[i] = (s, item)
            else:
                more = False
            while count in buf:
                s, item = buf[count]
                del buf[count]
                if s is not None:
                    record.write_idx(item[0], s)

                if count % 10000 == 0:
                    cur_time = time.time()
                    print("        ", count, "items saved in the RecordIO in", cur_time - pre_time, "secs")
                    pre_time = cur_time
                count += 1

    def _generate_rec_file(self, input_dir_path_str, lst_file_content_lst, output_file_prefix_str):
        """
        This method generates the RecordIO .rec file.

        Parameters
        ----------
        input_dir_path_str: string
                Path of the directory containing images to use.

        lst_file_content_lst: list of lists
                List of lists in the following format: [integer_image_index, path_to_image, label_index]

        output_file_prefix_str: string
                Prefix indicating where both .rec and .idx files will be written.

        Returns
        -------
        None
        """

        # Create queues for Producer-Consumer paradigm
        q_in = [mp.Queue(1024) for i in range(self.n_cores - 1)]
        q_out = mp.Queue(1024)
        
        # define the processes
        read_processes = [mp.Process(target = self._read_worker, args = (input_dir_path_str, q_in[i], q_out)) for i in range(self.n_cores - 1)]
        
        # process images with n_cores - 1 process
        for p in read_processes:
            p.start()
            
        # only use one process to write .rec to avoid race-condtion
        write_process = mp.Process(target = self._write_worker, args = (q_out, output_file_prefix_str))
        write_process.start()
        
        # put the image list into input queue
        for i, item in enumerate(lst_file_content_lst):
            q_in[i % len(q_in)].put((i, item))
        
        for q in q_in:
            q.put(None)
            
        for p in read_processes:
            p.join()

        q_out.put(None)
        write_process.join()

    def generate_record_io_file(self, pets_id_lst, output_file_prefix, input_images_dir):
        """
        This method generates the .rec RecordIO file and its .lst associated file.
        It also apply a transformation on each image.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        output_file_prefix: string
                Prefix indicating where both .rec and .idx files will be written.

        input_images_dir: string
                Path of the directory containing images to use.

        Returns
        -------
        None
        """

        st = time.time()
        print("Generating RecordIO file from images found in ", input_images_dir, "...") 

        # Generate the .lst file
        print("    Generating the .lst file as", output_file_prefix + ".lst", "...")
        lst_file_content_lst = self._generate_lst_file(pets_id_lst, input_images_dir, output_file_prefix + ".lst")

        # Generate the .rec file
        print("    Generating the .rec file as", output_file_prefix + ".rec", "...")
        self._generate_rec_file(input_images_dir, lst_file_content_lst, output_file_prefix)

        print("Generating RecordIO file from images... done in", round(time.time() - st, 3), "secs")

class ImagesFeaturesExtractionStep(object):
    """
    This class extracts the features from each image provided with pets data.
    """

    def __init__(self, neural_network_weights_dir):
        """
        This is the class' constructor.

        Parameters
        ----------
        neural_network_weights_dir: string
                Path where the weights of the MxNet model used to extract features
                can be found.

        Returns
        -------
        None
        """

        self.neural_network_weights_dir = neural_network_weights_dir
        self._context = [mx.gpu(i) for i in mx.test_utils.list_gpus()] if mx.test_utils.list_gpus() else mx.cpu()
        self._batch_size = 512
        self._constant_features_lst = None

        # Load the pretrained ResNet-50
        os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4" # Use 4 threads to load the data
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.neural_network_weights_dir + "resnet-50", 0)

        all_layers = sym.get_internals()
        sym3 = all_layers["flatten0_output"]
        self._model = mx.mod.Module(symbol = sym3, label_names = None, context = self._context)
        self._model.bind(for_training = False, data_shapes = [("data", (self._batch_size, 3, 224, 224))])
        self._model.set_params(arg_params, aux_params)

    def extract_features_from_images(self, pets_id_lst, recordio_dir_path_str):
        """
        This method is used to extract features from pets' images using a pretrained neural network.

        Parameters
        ----------
        pets_id_lst : list
                List containing the IDs of the pets we want to extract data for.

        recordio_dir_path_str : string
                Path of the directory containing the RecordIO files.
                
        Returns
        -------
        extracted_features_df: pd.DataFrame
                Data frame containing the extracted data.
        """

        st = time.time()
        print("Extracting features from images using Resnet-50 stored in RecordIO", recordio_dir_path_str, "...") 

        # Get images ID
        images_df = pd.read_csv(recordio_dir_path_str + ".lst", sep = "\t", header = None)
        images_df.columns = ["integer_image_index", "label_index", "path_to_image"]

        # Create the data iterator and make predictions
        data_iter = mx.io.ImageRecordIter(path_imglist = recordio_dir_path_str + ".lst", path_imgrec = recordio_dir_path_str + ".rec", path_imgidx = recordio_dir_path_str + ".idx", data_shape = (3, 224, 224), batch_size = self._batch_size, shuffle = False)
        predictions_npa = self._model.predict(data_iter).asnumpy()
        
        predictions_df = pd.DataFrame(predictions_npa, index = images_df["path_to_image"].tolist(), columns = ["resnet_output_" + str(c) for c in range(2048)])
        predictions_df = predictions_df.reset_index()
        predictions_df["PetID"] = predictions_df["index"].apply(lambda x: x.split("-")[0])
        predictions_df.drop("index", axis = 1, inplace = True)

        print("    Grouping extracted features by 'PetID'...")

        predictions_df = predictions_df.groupby(["PetID"]).agg(["mean", "std"])
        predictions_df.columns = pd.Index(["{}_{}_{}".format("img_features", c[0], c[1]) for c in predictions_df.columns.tolist()])

        # Remove missing values caused by std on constant series
        predictions_df.fillna(0, inplace = True)

        # Remove constant columns
        if self._constant_features_lst is None:
            cardinality_counts_sr = predictions_df.nunique()
            self._constant_features_lst = cardinality_counts_sr.loc[cardinality_counts_sr == 1].index.tolist()

        predictions_df.drop(self._constant_features_lst, axis = 1, inplace = True)

        print("Extracting features from images using Resnet-50... done in", round(time.time() - st, 3), "secs")

        return predictions_df     

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file contains the code needed for the preprocessing step.              #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-12-30                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.decomposition import PCA

class PreprocessingStep(BaseEstimator, TransformerMixin):
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

        self._stemmer = SnowballStemmer("english")
        self._highly_correlated_features_pca = PCA(n_components = 2)
        self._highly_correlated_features_lst = []

    def _detect_lang(self, x):
        try:
            return detect(str(x))
        except:
            return np.nan

    def _clean_text(self, sr):
        # Convert text to lowercase
        sr = sr.str.lower()

        # Remove numbers
        sr = sr.str.replace("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", "")

        # Remove punctuation
        sr = sr.str.replace(">|\-|,|:|<|\#|\||\$|\]|\^|\+|=|\(|_|\*|`|;|/|\)|\?|\&|\{|!|\.|%|@|\[|\~|\\|\"|\'|\}", "")

        # Remove non-letters or space chars
        sr = sr.str.replace("[^a-z\s]", "")

        # Remove large spaces
        sr = sr.str.replace("\s+", " ")

        # Stem words
        sr = sr.apply(lambda x: " ".join([self._stemmer.stem(w) for w in x.split(" ")]))

        # Remove less-than-two-chars words

        return sr
                                      
    def fit(self, X, y):
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
        
        # Get highly correlated features
        cormat = X.select_dtypes(np.number).corr()
        for i in range(cormat.shape[0]):
            for j in range(i + 1, cormat.shape[0]):
                if np.abs(cormat.iloc[i, j]) > 0.99:
                    self._highly_correlated_features_lst.append(cormat.columns[i])
                    self._highly_correlated_features_lst.append(cormat.columns[j])
                    
        self._highly_correlated_features_lst = list(set(self._highly_correlated_features_lst))

        X2 = X[self._highly_correlated_features_lst].fillna(0)
        self._highly_correlated_features_pca.fit(X2)

        return self
    
    def transform(self, X):
        """
        This method is called transform the data given in argument.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """
                            
        st = time.time()
        print("Preprocessing data...")   

        # Compute growth rate
        X["GrowthRate"] = X["Age"] / X["MaturitySize"]

        # Reorder some features to increase correlation with target
        X["Sterilized"] = X["Sterilized"].map({2: 1, 3: 2, 1: 3}) # After mapping: 1 = No, 2 = Not sure, 3 = Yes
        X["Dewormed"] = X["Dewormed"].map({2: 1, 1: 2, 3: 3}) # After mapping: 1 = No, 2 = Yes, 3 = Not sure
        X["Vaccinated"] = X["Vaccinated"].map({2: 1, 3: 2, 1: 3}) # After mapping: 1 = No, 2 = Not sure, 3 = Yes
        X["MaturitySize"] = X["MaturitySize"].map({4: 1, 1: 2, 3: 3, 2: 4})

        # Create features from Breeds
        ## Look for pure-breed animals
        X["pure_breed"] = ((X["Breed2"] == "NA") | (X["Breed1"] == X["Breed2"]) | (X["Breed1"].str.contains("Domestic|Mixed"))).astype(np.int8)
        X["Breed1_Mixed_Breed"] = (X["Breed1"] == "Mixed Breed").astype(np.int8)
        X["Breed2_Mixed_Breed"] = (X["Breed2"] == "Mixed Breed").astype(np.int8)
        X["Breed1_Domestic"] = (X["Breed1"].str.contains("Domestic")).astype(np.int8)
        X["Breed2_Domestic"] = (X["Breed2"].str.contains("Domestic")).astype(np.int8)

        # Create bins for 'Age'
        X["age_lt_3months"] = (X["Age"] < 3).astype(np.int8)
        
        # Convert 'Age' in years
        X["Age_years"] = X["Age"].apply(lambda x: x / 12 if x / 12 in [0.25, 0.50, 0.75] else int(x / 12))

        # Create features for Color
        X["nb_NA_colors"] = (X["Color1"] == "NA").astype(np.int8) + (X["Color2"] == "NA").astype(np.int8) + (X["Color3"] == "NA").astype(np.int8)
        X["Colors"] = X["Color1"].astype(str) + " " + X["Color2"].astype(str) + " " + X["Color3"]

        # Look for free animals
        X["free_animal"] = (X["Fee"] == 0).astype(np.int8)
        
        # Make 'Type' a binary variable
        X["Type"] = (X["Type"] - 1).astype(np.int8)

        # Create features based on veterinary-related fields
        X["veterinary_nb_yes"] = (X["Vaccinated"] == 3).astype(np.int8) + (X["Dewormed"] == 2).astype(np.int8) + (X["Sterilized"] == 3).astype(np.int8)
        X["veterinary_nb_no"] = (X["Vaccinated"] == 1).astype(np.int8) + (X["Dewormed"] == 1).astype(np.int8) + (X["Sterilized"] == 1).astype(np.int8)
        #X["veterinary_nb_not_sure"] = (X["Vaccinated"] == "Not sure").astype(np.int8) + (X["Dewormed"] == "Not sure").astype(np.int8) + (X["Sterilized"] == "Not sure").astype(np.int8)
        X["veterinary_all_yes"] = (X["veterinary_nb_yes"] == 3).astype(np.int8)

        # Remove missing values from text columns
        for col in ["Name", "Description", "metadata_annots_top_desc", "sentiment_file_entities_name_lst", "sentiment_file_entities_type_lst"]:
            X[col] = X[col].fillna("")

        # Create statistics based on description and name
        numbers_re = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
        words_re = re.compile("^[a-z]+$")

        for col in ["Name", "Description"]:
            X[col + "_nb_chars"] = X[col].str.len()
            X[col + "_nb_tokens"] = X[col].apply(lambda x: len(x.lower().split(" ")))
            X[col + "_nb_words"] = X[col].apply(lambda x: len([w for w in x.lower().split(" ") if bool(words_re.match(w))]))
            X[col + "_nb_numbers"] = X[col].apply(lambda x: len([w for w in x.lower().split(" ") if bool(numbers_re.match(w))]))
            X[col + "_nb_letters"] = X[col].apply(lambda x: len(re.findall("[a-zA-Z]", x)))
            X[col + "_nb_digits"] = X[col].apply(lambda x: len(re.findall("[0-9]", x)))
            X[col + "_nb_capital_letters"] = X[col].apply(lambda x: len(re.findall("[A-Z]", x)))
            X[col + "_nb_punctuation"] = X[col].apply(lambda x: len(re.findall(">|\-|,|:|<|\#|\||\$|\]|\^|\+|=|\(|_|\*|`|;|/|\)|\?|\&|\{|!|\.|%|@|\[|\~|\\|\"|\'|\}", x)))
            X[col + "_nb_garbage"] = X[col + "_nb_chars"] - (X[col + "_nb_letters"] + X[col + "_nb_digits"] + X[col + "_nb_punctuation"])
            X[col + "_digits_ratio"] = X[col + "_nb_digits"] / X[col + "_nb_chars"]
            X[col + "_letters_ratio"] = X[col + "_nb_letters"] / X[col + "_nb_chars"]
            X[col + "_capital_letters_ratio"] = X[col + "_nb_capital_letters"] / X[col + "_nb_chars"]
            X[col + "_capital_letters_to_letters_ratio"] = X[col + "_nb_capital_letters"] / X[col + "_nb_letters"]
            X[col + "_punctuation_ratio"] = X[col + "_nb_punctuation"] / X[col + "_nb_chars"]

        X["total_nb_chars"] = X["Name_nb_chars"] + X["Description_nb_chars"]
        X["total_nb_tokens"] = X["Name_nb_tokens"] + X["Description_nb_tokens"]
        X["total_nb_words"] = X["Name_nb_words"] + X["Description_nb_words"]
        X["total_nb_numbers"] = X["Name_nb_numbers"] + X["Description_nb_numbers"]
        X["total_nb_letters"] = X["Name_nb_letters"] + X["Description_nb_letters"]
        X["total_nb_digits"] = X["Name_nb_digits"] + X["Description_nb_digits"]
        X["total_nb_capital_letters"] = X["Name_nb_capital_letters"] + X["Description_nb_capital_letters"]
        X["total_nb_punctuation"] = X["Name_nb_punctuation"] + X["Description_nb_punctuation"]
        X["total_nb_garbage"] = X["Name_nb_garbage"] + X["Description_nb_garbage"]
        X["total_digits_ratio"] = X["total_nb_digits"] / X["total_nb_chars"]
        X["total_letters_ratio"] = X["total_nb_letters"] / X["total_nb_chars"]
        X["total_capital_letters_ratio"] = X["total_nb_capital_letters"] / X["total_nb_chars"]
        X["total_capital_letters_to_letters_ratio"] = X["total_nb_capital_letters"] / X["total_nb_letters"]
        X["total_punctuation_ratio"] = X["total_nb_punctuation"] / X["total_nb_letters"]

        # Count smileys in description
        ## Data comes from https://demos.emojione.com/latest/ascii-smileys.html and https://github.com/words/emoji-emotion/blob/master/index.json
        ## Source of data added in https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/75943
        ## smileys_dict's keys are smileys and smileys_dict's values are corresponding smiley's polarity (i.e. emotion associated to smiley, either positive or negative)
        smileys_dict = {'<3': 3, '</3': -3, ":')": 3, ":'-)": 3, ':D': 2, ':-D': 2, '=D': 2, ':)': 1, ':-)': 1, '=]': 1, '=)': 1, ':]': 1, "':)": 2, "':-)": 2, "'=)": 2, "':D": 2, 
                        "':-D": 2, "'=D": 2, '>:)': 1, '>;)': 1, '>:-)': 1, '>=)': 1, ';)': 3, ';-)': 3, '*-)': 3, '*)': 3, ';-]': 3, ';]': 3, ';D': 3, ';^)': 3, "':(": -1, "':-(": -1, 
                        "'=(": -1, ':*': 3, ':-*': 3, '=*': 3, ':^*': 3, '>:P': -1, 'X-P': -1, 'x-p': -1, '>:[': -2, ':-(': -2, ':(': -2, ':-[': -2, ':[': -2, '=(': -2, '>:(': -3, 
                        '>:-(': -3, ':@': -3, ":'(": -2, ":'-(": -2, ';(': -2, ';-(': -2, '>.<': -2, 'D:': -2, ':$': -2, '=$': -2, '#-)': -1, '#)': -1, '%-)': -1, '%)': -1, 'X)': -1, 
                        'X-)': -1, 'O:-)': 3, '0:-3': 3, '00:03': 3, '0:-)': 3, '0:)': 3, '0;^)': 3, 'O:)': 3, 'O;-)': 3, 'O=)': 3, '0;-)': 3, 'O:-3': 3, 'O:3': 3, 'B-)': 1, 'B)': 1, 
                        '8)': 1, '8-)': 1, 'B-D': 1, '8-D': 1, '-_-': 0, '-__-': 0, '-___-': 0, '>:\\': -2, '>:/': -2, ':-/': -2, ':-.': -2, ':/': -2, ':\\': -2, '=/': -2, '=\\': -2, 
                        ':L': -2, '=L': -2, ':P': 1, ':-P': 1, '=P': 1, ':-p': 1, ':p': 1, '=p': 1, ':-Þ': 1, ':Þ': 1, ':þ': 1, ':-þ': 1, ':-b': 1, ':b': 1, 'd:': 1, ':-O': -2, ':O': -2, 
                        ':-o': -2, ':o': -2, 'O_O': -2, '>:O': -2, ':-X': 0, ':X': 0, ':-#': 0, ':#': 0, '=X': 0, '=x': 0, ':x': 0, ':-x': 0, '=#': 0}
        smileys_re_str = "|".join([re.escape(s) for s in smileys_dict.keys()])
        description_smileys_sr = X["Description"].str.findall(smileys_re_str)
        X["smileys_count"] = description_smileys_sr.str.len()
        X["smileys_total_polarity"] = description_smileys_sr.apply(lambda x: np.sum([smileys_dict[s] for s in x]))
        X["smileys_polarity_variance"] = description_smileys_sr.apply(lambda x: np.std([smileys_dict[s] for s in x]))

        """unicode_smileys_df = pd.read_csv(EMOJI_SENTIMENT_DATA_str)
        unicode_smileys_df = unicode_smileys_df[["Emoji", "Negative", "Neutral", "Positive"]]
        unicode_smileys_df["count"] = unicode_smileys_df["Negative"] + unicode_smileys_df["Neutral"] + unicode_smileys_df["Positive"]
        unicode_smileys_df["Negative"] = unicode_smileys_df["Negative"] / unicode_smileys_df["count"]
        unicode_smileys_df["Neutral"] = unicode_smileys_df["Neutral"] / unicode_smileys_df["count"]
        unicode_smileys_df["Positive"] = unicode_smileys_df["Positive"] / unicode_smileys_df["count"]
        description_smileys_sr = X["Description"].str.findall("|".join(unicode_smileys_df["Emoji"].tolist()))"""

        # Count the number of important keywords in description
        for col in ["whatsapp", "contact", "adopted", "home", "free", "loving", "adoption", "adopt", "friendly", "friend", "interested", "healthy", "neutered", "food"]:
            X["Description_nb_" + col + "_count"] = X["Description"].str.lower().str.count(col)

        # Count the number of important keywords in metadata_annots_top_desc
        for col in ["puppy", "street_dog", "carnivoran", "companion_dog", "fauna", "kitten", "dog_breed_group", "snout", "whiskers", "domestic_long_haired_cat", "turkish_angora", "black_cat"]:
            X["metadata_annots_top_desc_" + col + "_count"] = X["metadata_annots_top_desc"].str.lower().str.count(col)
               
        X["metadata_annots_top_desc_len"] = X["metadata_annots_top_desc"].str.split(" ").str.len()

        # Count RescuerID occurrences:
        rescuer_count = X.groupby(["RescuerID"])["PetID"].count().reset_index()
        rescuer_count.columns = ["RescuerID", "RescuerID_count"]
        X = X.merge(rescuer_count, how = "left", on = "RescuerID")

        # RescuerID stats based on Name and Description:
        X2 = X[["RescuerID", "Description_nb_chars", "Description_nb_words", "Description_nb_punctuation", "Description_nb_garbage", "Description_letters_ratio", "Description_punctuation_ratio"]]
        tmp = X2.groupby(["RescuerID"])[["Description_nb_chars", "Description_nb_words", "Description_nb_punctuation", "Description_nb_garbage", "Description_letters_ratio", "Description_punctuation_ratio"]].agg(["mean", "std"]).reset_index()
        tmp.columns = ["RescuerID", "RescuerID_Description_avg_nb_chars", "RescuerID_Description_std_nb_chars", "RescuerID_Description_avg_nb_words", "RescuerID_Description_std_nb_words", 
                                 "RescuerID_Description_avg_nb_punctuation", "RescuerID_Description_std_nb_punctuation", "RescuerID_Description_avg_nb_garbage", "RescuerID_Description_std_nb_garbage",
                                 "RescuerID_Description_avg_letters_ratio", "RescuerID_Description_std_letters_ratio", "RescuerID_Description_avg_punctuation_ratio", "RescuerID_Description_std_punctuation_ratio"]
        tmp.fillna(0, inplace = True)
        X = X.merge(tmp, how = "left", on = "RescuerID")

        X2 = X[["RescuerID", "Name_nb_chars", "Name_nb_words", "Name_nb_punctuation", "Name_nb_garbage", "Name_letters_ratio", "Name_punctuation_ratio"]]
        tmp = X2.groupby(["RescuerID"])[["Name_nb_chars", "Name_nb_words", "Name_nb_punctuation", "Name_nb_garbage", "Name_letters_ratio", "Name_punctuation_ratio"]].agg(["mean", "std"]).reset_index()
        tmp.columns = ["RescuerID", "RescuerID_Name_avg_nb_chars", "RescuerID_Name_std_nb_chars", "RescuerID_Name_avg_nb_words", "RescuerID_Name_std_nb_words", 
                                 "RescuerID_Name_avg_nb_punctuation", "RescuerID_Name_std_nb_punctuation", "RescuerID_Name_avg_nb_garbage", "RescuerID_Name_std_nb_garbage",
                                 "RescuerID_Name_avg_letters_ratio", "RescuerID_Name_std_letters_ratio", "RescuerID_Name_avg_punctuation_ratio", "RescuerID_Name_std_punctuation_ratio"]
        tmp.fillna(0, inplace = True)
        X = X.merge(tmp, how = "left", on = "RescuerID")

        """
        # Clean description
        X["Description"] = self._clean_text(X["Description"])

        # Clean language
        X["lang"] = X["Description"].apply(self._detect_lang)
        X["Description_language"] = X["sentiment_file_language"].apply(lambda x: x if x == "en" else np.nan)
        X["Description_language"].loc[(X["Description_language"].isnull()) & ((X["sentiment_file_language"].str.contains("zh")) | (X["lang"].str.contains("zh")))] = "zh"
        X["Description_language"].fillna("id", inplace = True)"""

        # Count number of animals of the same breed in each state
        tmp = X[["Breed1", "State", "Quantity"]].groupby(["Breed1", "State"]).sum().reset_index()
        tmp.columns = ["Breed1", "State", "Breed1_quantity_by_State"]
        X = X.merge(tmp, how = "left", on = ["Breed1", "State"])

        # Count number of animals of the same breed
        tmp = X[["Breed1", "Quantity"]].groupby(["Breed1"]).sum().reset_index()
        tmp.columns = ["Breed1", "quantity_by_Breed1"]
        X = X.merge(tmp, how = "left", on = ["Breed1"])

        # Count number of animals of the same breed and gender
        tmp = X[["Breed1", "Gender", "Quantity"]].groupby(["Breed1", "Gender"]).sum().reset_index()
        tmp.columns = ["Breed1", "Gender", "quantity_by_Breed1_and_Gender"]
        X = X.merge(tmp, how = "left", on = ["Breed1", "Gender"])
        
        # Generate features about Pet's name
        X["No_Name"] = X["Name"].isnull() | X["Name"].str.contains("No\sName|No\sName\sYet|Kittens|Kitten|Puppies|Nameless|Unnamed")
        X["No_Name"] = X["No_Name"] | ~(X["Name"].str.contains("^[a-zA-Z\s]+$").fillna(False))
        X["No_Name"] = X["No_Name"] | ((X["Name"].str.len() < 4).fillna(True))
        X["No_Name"] = X["No_Name"].astype(np.int8)

        # Look for key words in 'Description' and 'Name'
        X["adopted_count"] = X["Description"].str.lower().str.count("adopted") + X["Name"].str.lower().str.count("adopted")

        X["Single"] = (X["Quantity"] == 1).astype(np.int8)

        # Adding data about States
        # GDP by state (in million RM): https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
        state_gdp = {
            "Johor": 116.679,
            "Kedah": 40.596,
            "Kelantan": 23.02,
            "Kuala Lumpur": 190.075,
            "Labuan": 5.984,
            "Melaka": 37.274,
            "Negeri Sembilan": 42.389,
            "Pahang": 52.452,
            "Perak": 67.629,
            "Perlis": 5.642,
            "Pulau Pinang": 81.284,
            "Sabah": 80.167,
            "Sarawak": 121.414,
            "Selangor": 280.698,
            "Terengganu": 32.270
        }

        # state population: https://en.wikipedia.org/wiki/Malaysia
        state_population = {
            "Johor": 33.48283,
            "Kedah": 19.47651,
            "Kelantan": 15.39601,
            "Kuala Lumpur": 16.74621,
            "Labuan": 0.86908,
            "Melaka": 8.21110,
            "Negeri Sembilan": 10.21064,
            "Pahang": 15.00817,
            "Perak": 23.52743,
            "Perlis": 2.31541,
            "Pulau Pinang": 15.61383,
            "Sabah": 32.06742,
            "Sarawak": 24.71140,
            "Selangor": 54.62141,
            "Terengganu": 10.35977
        }

        X["state_gdp"] = X["State"].map(state_gdp)
        X["state_population"] = X["State"].map(state_population)

        # Compress highly correlated features
        X2 = X[self._highly_correlated_features_lst].fillna(0)
        X2 = pd.DataFrame(self._highly_correlated_features_pca.transform(X2), index = X.index, columns = ["highly_correlated_features_PCA_comp_1", "highly_correlated_features_PCA_comp_2"])
        X.drop(self._highly_correlated_features_lst, axis = 1, inplace = True)
        X = pd.concat([X, X2], axis = 1)
        
        # Merge Name and Description
        X["Description"] = X["Name"].astype(str) + " " + X["Description"]
        
        # Drop columns
        #X.drop(["Name", "RescuerID", "PetID", "lang", "sentiment_file_language"], axis = 1, inplace = True)
        X.drop(["Name", "RescuerID", "PetID", "sentiment_file_language"], axis = 1, inplace = True)

        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X

# File paths
TRAIN_DATA_str = "../input/petfinder-adoption-prediction/train/train.csv"
TEST_DATA_str = "../input/petfinder-adoption-prediction/test/test.csv"

BREEDS_DATA_str = "../input/petfinder-adoption-prediction/breed_labels.csv"
COLORS_DATA_str = "../input/petfinder-adoption-prediction/color_labels.csv"
STATES_DATA_str = "../input/petfinder-adoption-prediction/state_labels.csv"

# Images
TRAIN_IMAGES_DIR_str = "../input/petfinder-adoption-prediction/train_images/"
TEST_IMAGES_DIR_str = "../input/petfinder-adoption-prediction/test_images/"
TRAIN_RECORDIO_PREFIX_str = "../input/train_recordio"
TEST_RECORDIO_PREFIX_str = "../input/test_recordio"

# Sentiment-related data
TRAIN_SENTIMENT_DATA_DIR_str = "../input/petfinder-adoption-prediction/train_sentiment/"
TEST_SENTIMENT_DATA_DIR_str = "../input/petfinder-adoption-prediction/test_sentiment/"

# Metadata files
TRAIN_METADATA_DIR_str = "../input/petfinder-adoption-prediction/train_metadata/"
TEST_METADATA_DIR_str = "../input/petfinder-adoption-prediction/test_metadata/"

# External data
CAT_AND_DOG_BREEDS_DATA_str = "../input/cat-and-dog-breeds-parameters/rating.json" # https://www.kaggle.com/hocop1/cat-and-dog-breeds-parameters
NN_WEIGHTS_str = "../input/resnet50-weights-for-mxnet/"

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file provides everything needed to load the data.                      #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-12-30                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def load_data(training_set_path_str, testing_set_path_str, breeds_data_path_str, states_data_path_str, colors_data_path_str, cat_and_dog_breeds_data_path_str, enable_validation, target_name_str):
    """
    This function is a wrapper for the loading of the data.

    Parameters
    ----------
    training_set_path_str : string
            A string containing the path of the training set file.

    testing_set_path_str : string
            A string containing the path of the testing set file.

    breeds_data_path_str : string
            A string containing the path of the file containing breeds matchings.

    states_data_path_str : string
            A string containing the path of the file containing states matchings.

    colors_data_path_str : string
            A string containing the path of the file containing colors matchings.

    cat_and_dog_breeds_data_path_str : string
            A string containing the path of the file ratings.json taken from https://www.kaggle.com/hocop1/cat-and-dog-breeds-parameters/version/1

    enable_validation : boolean
            A boolean indicating if we are validating our model or if we are creating a submission for Kaggle.

    target_name_str : string
            A string indicating the target column name.
            
    Returns
    -------
    training_set_df : pd.DataFrame
            A pandas DataFrame containing the training set.

    testing_set_df : pd.DataFrame
            A pandas DataFrame containing the testing set.

    breeds_data_df : pd.DataFrame
            A pandas DataFrame containing the breeds matchings.

    states_data_df : pd.DataFrame
            A pandas DataFrame containing the states matchings.

    colors_data_df : pd.DataFrame
            A pandas DataFrame containing the colors matchings.
            
    target_sr : pd.Series
            The target values for the training part.

    truth_sr : pd.Series
            The target values for the validation part.
    """

    # Load the data
    print("Loading the data...")
    # Load train and test data
    training_set_df = pd.read_csv(training_set_path_str)
    testing_set_df = pd.read_csv(testing_set_path_str)

    # Loading data associated with breeds, states and colors
    breeds_data_df = pd.read_csv(breeds_data_path_str, index_col = 0)
    breeds_data_df.drop("Type", axis = 1, inplace = True)
    states_data_df = pd.read_csv(states_data_path_str, index_col = 0)
    colors_data_df = pd.read_csv(colors_data_path_str, index_col = 0)

    breeds_data_dict = breeds_data_df.to_dict()["BreedName"]
    states_data_dict = states_data_df.to_dict()["StateName"]
    colors_data_dict = colors_data_df.to_dict()["ColorName"]
    
    # Merging breeds, states and colors with main data
    training_set_df["Breed1"] = training_set_df["Breed1"].map(breeds_data_dict, na_action = "ignore")
    training_set_df["Breed2"] = training_set_df["Breed2"].map(breeds_data_dict, na_action = "ignore")
    training_set_df["Color1"] = training_set_df["Color1"].map(colors_data_dict, na_action = "ignore")
    training_set_df["Color2"] = training_set_df["Color2"].map(colors_data_dict, na_action = "ignore")
    training_set_df["Color3"] = training_set_df["Color3"].map(colors_data_dict, na_action = "ignore")
    training_set_df["State"] = training_set_df["State"].map(states_data_dict, na_action = "ignore")
    
    testing_set_df["Breed1"] = testing_set_df["Breed1"].map(breeds_data_dict, na_action = "ignore")
    testing_set_df["Breed2"] = testing_set_df["Breed2"].map(breeds_data_dict, na_action = "ignore")
    testing_set_df["Color1"] = testing_set_df["Color1"].map(colors_data_dict, na_action = "ignore")
    testing_set_df["Color2"] = testing_set_df["Color2"].map(colors_data_dict, na_action = "ignore")
    testing_set_df["Color3"] = testing_set_df["Color3"].map(colors_data_dict, na_action = "ignore")
    testing_set_df["State"] = testing_set_df["State"].map(states_data_dict, na_action = "ignore")

    # Remove missing values created by merge
    for col in ["Breed1", "Breed2", "Color1", "Color2", "Color3", "State"]:
        training_set_df[col].fillna("NA", inplace = True)
        testing_set_df[col].fillna("NA", inplace = True)

    # Loading external data (species ratings)
    with open(cat_and_dog_breeds_data_path_str, "r") as f:
        cat_and_dog_breeds_data_dict = json.load(f)

    cat_breeds_data_dict = cat_and_dog_breeds_data_dict["cat_breeds"]
    dog_breeds_data_dict = cat_and_dog_breeds_data_dict["dog_breeds"]
       
    ## Get all features names
    all_keys_lst = []
    for k, v in cat_breeds_data_dict.items():
        all_keys_lst.extend(list(v.keys()))

    for k, v in dog_breeds_data_dict.items():
        all_keys_lst.extend(list(v.keys()))

    ## Remove duplicates
    all_keys_lst = list(set(all_keys_lst))
    all_keys_lst = list(set([s.strip().lower().replace(" ", "_") for s in all_keys_lst]))

    ## Build the DataFrame
    breeds_ratings_dict = {"Breed1": []}
    for k in all_keys_lst:
        breeds_ratings_dict[k] = []

    for k, v in cat_breeds_data_dict.items():
        breeds_ratings_dict["Breed1"].append(k)

        for k2 in all_keys_lst:
            cleaned_keys_lst = [s.strip().lower().replace(" ", "_") for s in v.keys()]
            if k2 in cleaned_keys_lst:
                idx = cleaned_keys_lst.index(k2)
                breeds_ratings_dict[k2].append(v[list(v.keys())[idx]])
            else:
                breeds_ratings_dict[k2].append(0)
                
    for k, v in dog_breeds_data_dict.items():
        breeds_ratings_dict["Breed1"].append(k)

        for k2 in all_keys_lst:
            cleaned_keys_lst = [s.strip().lower().replace(" ", "_") for s in v.keys()]
            if k2 in cleaned_keys_lst:
                idx = cleaned_keys_lst.index(k2)
                breeds_ratings_dict[k2].append(v[list(v.keys())[idx]])
            else:
                breeds_ratings_dict[k2].append(0)

    breeds_ratings_df = pd.DataFrame(breeds_ratings_dict)
    breeds_ratings_df.columns = ["breeds_ratings_" + c if c != "Breed1" else c for c in breeds_ratings_df.columns]
    
    training_set_df = training_set_df.merge(breeds_ratings_df, how = "left", on = "Breed1")
    testing_set_df = testing_set_df.merge(breeds_ratings_df, how = "left", on = "Breed1")
        
    # Generate a validation set if enable_validation is True
    if enable_validation:
        print("Generating validation set...")
        test_size_ratio = 0.2

        # Split data on 'RescuerID' feature as this feature is not overlapping train and test
        unique_rescuer_ids_npa = training_set_df["RescuerID"].unique()
        train_rescuer_ids_npa, test_rescuer_ids_npa = train_test_split(unique_rescuer_ids_npa, test_size = test_size_ratio, random_state = 2019)

        testing_set_df = training_set_df.loc[training_set_df["RescuerID"].isin(test_rescuer_ids_npa)]
        training_set_df = training_set_df.loc[training_set_df["RescuerID"].isin(train_rescuer_ids_npa)]
    
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

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This is the entry point of the solution.                                    #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-12-30                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import pickle
import gc
import seaborn as sns
import matplotlib.pyplot as plt

import json

import scipy as sp
import pandas as pd
import numpy as np

from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from collections import Counter

import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import cohen_kappa_score
from ml_metrics import quadratic_weighted_kappa

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
    
    st = time.time()

    enable_validation = False

    print("Loading data...")
    training_data_df, testing_data_df, target_sr, truth_sr = load_data(TRAIN_DATA_str, TEST_DATA_str, BREEDS_DATA_str, STATES_DATA_str, COLORS_DATA_str, CAT_AND_DOG_BREEDS_DATA_str, enable_validation, "AdoptionSpeed")

    # Extracting IDs
    train_id = training_data_df["PetID"]
    test_id = testing_data_df["PetID"]

    print("training_data_df.shape:", training_data_df.shape)
    print("testing_data_df.shape:", testing_data_df.shape)
    
    print("Data loaded in:", time.time() - st, "secs")
    
    # Extract data from sentiment files
    sfps = SentimentFilesPreprocessingStep(n_cores = -1)
    train_sentiment_desc_df, train_sentiments_df = sfps.extract_data_from_sentiment_files(train_id.tolist(), TRAIN_SENTIMENT_DATA_DIR_str)

    if enable_validation:
        test_sentiment_desc_df, test_sentiments_df = sfps.extract_data_from_sentiment_files(test_id.tolist(), TRAIN_SENTIMENT_DATA_DIR_str)
    else:
        test_sentiment_desc_df, test_sentiments_df = sfps.extract_data_from_sentiment_files(test_id.tolist(), TEST_SENTIMENT_DATA_DIR_str)

    # Extract data from metadata files
    mfps = MetadataFilesPreprocessingStep(n_cores = -1)
    train_metadata_desc_df, train_metadata_df = mfps.extract_data_from_metadata_files(train_id.tolist(), TRAIN_METADATA_DIR_str)

    if enable_validation:
        test_metadata_desc_df, test_metadata_df = mfps.extract_data_from_metadata_files(test_id.tolist(), TRAIN_METADATA_DIR_str)
    else:
        test_metadata_desc_df, test_metadata_df = mfps.extract_data_from_metadata_files(test_id.tolist(), TEST_METADATA_DIR_str)
    
    # Merging sentiment and metadata to main DataFrames
    training_data_df = training_data_df.merge(train_sentiments_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_metadata_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_metadata_desc_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_sentiment_desc_df, how = "left", on = "PetID")

    testing_data_df = testing_data_df.merge(test_sentiments_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_metadata_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_metadata_desc_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_sentiment_desc_df, how = "left", on = "PetID")
    
    # Extract features from images
    imes = ImagesMetadataExtractionStep(n_cores = -1)
    train_images_metadata_df = imes.extract_metadata_from_images(train_id.tolist(), TRAIN_IMAGES_DIR_str)

    if enable_validation:
        test_images_metadata_df = imes.extract_metadata_from_images(test_id.tolist(), TRAIN_IMAGES_DIR_str)
    else:
        test_images_metadata_df = imes.extract_metadata_from_images(test_id.tolist(), TEST_IMAGES_DIR_str)

    training_data_df = training_data_df.merge(train_images_metadata_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_images_metadata_df, how = "left", on = "PetID")

    # Create a RecordIO file to load images faster
    gmrf = GenerateMxNetRecordIOFile(n_cores = 5)
    gmrf.generate_record_io_file(train_id.tolist(), TRAIN_RECORDIO_PREFIX_str, TRAIN_IMAGES_DIR_str)

    if enable_validation:
        gmrf.generate_record_io_file(test_id.tolist(), TEST_RECORDIO_PREFIX_str, TRAIN_IMAGES_DIR_str)
    else:
        gmrf.generate_record_io_file(test_id.tolist(), TEST_RECORDIO_PREFIX_str, TEST_IMAGES_DIR_str)

    # Extract features from images (inside the RecordIO)
    ifes = ImagesFeaturesExtractionStep(NN_WEIGHTS_str)
    train_images_features_df = ifes.extract_features_from_images(train_id.tolist(), TRAIN_RECORDIO_PREFIX_str)

    if enable_validation:
        test_images_features_df = ifes.extract_features_from_images(test_id.tolist(), TEST_RECORDIO_PREFIX_str)
    else:
        test_images_features_df = ifes.extract_features_from_images(test_id.tolist(), TEST_RECORDIO_PREFIX_str)
    
    del gmrf, ifes
    gc.collect()

    pca = PCA(64)
    train_images_features_df = pd.DataFrame(pca.fit_transform(train_images_features_df), index = train_images_features_df.index, columns = ["images_features_PCA_" + str(i) for i in range(64)])
    test_images_features_df = pd.DataFrame(pca.transform(test_images_features_df), index = test_images_features_df.index, columns = ["images_features_PCA_" + str(i) for i in range(64)])
    train_images_features_df = train_images_features_df.reset_index()
    test_images_features_df = test_images_features_df.reset_index()
    training_data_df = training_data_df.merge(train_images_features_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_images_features_df, how = "left", on = "PetID")

    gc.collect()

    ## LightGBM hyperparameters
    lgb_params = {"application": "multiclass",
                  "boosting": "gbdt",
                  "metric": "qwk",
                  "num_class": 5,
                  "num_leaves": 70,
                  "max_depth": -1,
                  "learning_rate": 0.005,
                  "bagging_fraction": 0.95,
                  "feature_fraction": 0.15,
                  "min_split_gain": 0.02,
                  "min_child_samples": 140,
                  "min_child_weight": 0.02,
                  "verbosity": -1,
                  "data_random_seed": 17,
                  "nthread": 2,
                  "device": "cpu"}
        
    categorical_columns_to_be_encoded_lst = ["State", 
                                             "Breed1", "Breed2",
                                             "Color1", "Color2", "Color3"]#, "Description_language"]
    categorical_encoders_lst = [LabelBinarizer(), 
                                GroupingEncoder(encoder = TargetAvgEncoder(), threshold = 103), GroupingEncoder(encoder = TargetAvgEncoder(), threshold = 66),
                                LabelBinarizer(), LabelBinarizer(), LabelBinarizer()#, TargetAvgEncoder()
                                ]

    # Columns that will be encoded using SparseTextEncoder class
    lsa_parameters = {"min_df": 3,  "max_features": 10000, "strip_accents": "unicode", "analyzer": "word", "token_pattern": r"\w{1,}", "ngram_range": (1, 3), "use_idf": 1, "smooth_idf": 1, 
                      "sublinear_tf": 1, "stop_words": "english"}
    text_columns_to_be_encoded_lst = ["Description", "metadata_annots_top_desc", 
                                      "sentiment_file_entities_name_lst", "sentiment_file_entities_type_lst", "Colors"]
    text_encoders_lst = [LSAVectorizer(lsa_components = 140, tfidf_parameters = lsa_parameters), LSAVectorizer(lsa_components = 40, tfidf_parameters = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10, "token_pattern": r"[a-zA-Z_]+"}), 
                         LSAVectorizer(lsa_components = 20), TfidfVectorizer(), CountVectorizer()]
    
    # Put EfficientPipeline instead of Pipeline
    main_pipeline = Pipeline([
                              ("PreprocessingStep", PreprocessingStep()),
                              ("CategoricalFeaturesEncoder", CategoricalFeaturesEncoder(categorical_columns_to_be_encoded_lst, categorical_encoders_lst)),
                              ("SparseTextEncoder", SparseTextEncoder(text_columns_to_be_encoded_lst, text_encoders_lst, output_format = "pandas")),
                              ("DuplicatedFeaturesRemover", DuplicatedFeaturesRemover()),
                              ("BlendedLightGBM", BlendedLGBMClassifier(lgb_params, early_stopping_rounds = 150, eval_size = 0.2, eval_split_type = "random", verbose_eval = 100, nrounds = 10000))
                             ])
        
    # Train the model
    main_pipeline.fit(training_data_df, target_sr)
    
    # Make predictions
    predictions_npa = main_pipeline.predict(testing_data_df)
    
    # Evaluate the model
    print("Predicted Counts = ", Counter(predictions_npa))
    submission = pd.DataFrame({"PetID": test_id, "AdoptionSpeed": predictions_npa})
    submission["AdoptionSpeed"] = submission["AdoptionSpeed"].astype(np.int32)
    submission.to_csv("submission.csv", index = False)
    
    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")