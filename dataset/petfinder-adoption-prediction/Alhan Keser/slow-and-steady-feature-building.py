'''
If you find this useful, please give a thumbs up!

Thanks!
- Claire & Alhan

https://github.com/alhankeser/kaggle-petfinder
'''

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import make_scorer
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
import scipy.stats as stats
import math
import time
import traceback
import warnings
import os
import zipfile
import shutil
# import sys
import json

# Options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
warnings.filterwarnings(action="ignore")


class Explore:

    def get_dtype(cls, include_type=[], exclude_type=[]):
        df = cls.get_df('train')
        df.drop(columns=[cls.target_col], inplace=True)
        return df.select_dtypes(include=include_type, exclude=exclude_type)

    def get_non_numeric(cls):
        return cls.get_dtype(exclude_type=['float64', 'int', 'float32'])

    def get_numeric(cls):
        return cls.get_dtype(exclude_type=['object', 'category'])

    def get_categorical(cls, as_df=False):
        return cls.get_dtype(include_type=['object'])

    def get_correlations(cls, method='spearman'):
        df = cls.get_df('train')
        corr_mat = df.corr(method=method)
        corr_mat.sort_values(cls.target_col, inplace=True)
        corr_mat.drop(cls.target_col, inplace=True)
        return corr_mat[[cls.target_col]]

    def get_skewed_features(cls, df, features, skew_threshold=0.4):
        feat_skew = pd.DataFrame(
                    {'skew': df[features].apply(lambda x: stats.skew(x))})
        skewed = feat_skew[abs(feat_skew['skew']) > skew_threshold].index
        return skewed.values

    def show_boxplot(cls, x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    def plot_categorical(cls, df, cols):
        target = cls.target_col
        categorical = pd.melt(df, id_vars=[target],
                              value_vars=cols)
        grouped = categorical.groupby(['value', 'variable'],
                                      as_index=False)[target]\
            .mean().rename(columns={target: target + '_Mean'})
        categorical = pd.merge(categorical, grouped, how='left',
                               on=['variable', 'value'])\
            .sort_values(target + '_Mean')
        facet_grid = sns.FacetGrid(categorical, col="variable",
                                   col_wrap=3, size=5,
                                   sharex=False, sharey=False,)
        facet_grid = facet_grid.map(cls.show_boxplot, "value", target)
        plt.savefig('boxplots.png')


class Clean:

    def sample_ros(cls, df):
        if df.name == 'train':
            X = df.drop(cls.target_col, axis=1)
            y = df[cls.target_col]
            ros = RandomOverSampler(sampling_strategy='minority',
                                    random_state=1)
            X_ros, y_ros = ros.fit_sample(X, y)
            df = pd.DataFrame(list(X_ros),
                              columns=df.drop(cls.target_col, axis=1)
                              .columns)
            df[cls.target_col] = list(y_ros)
        return df

    def sample(cls, df, target_val_sets):
        if df.name == 'train':
            for target_val_set in target_val_sets:
                df_class_0 = df[df[cls.target_col] == target_val_set[0]]
                count_1 = df[cls.target_col].value_counts()[target_val_set[1]]
                df_class_0_sampled = df_class_0.sample(count_1,
                                                       replace='True',
                                                       random_state=1)
                df = pd.merge(df.drop(df_class_0.index),
                              df_class_0_sampled, how='outer')
        return df

    def keep_only_keep(cls, df):
        to_drop = set(df.columns.values) - set(cls.keep)
        if df.name == 'train':
            to_drop = to_drop - set([cls.target_col])
        to_drop = list(to_drop)
        df.drop(to_drop, axis=1, inplace=True)
        return df

    def remove_outliers(cls, df):
        if df.name == 'train':
            # GrLivArea (1299 & 524)
            # df.drop(df[(df['GrLivArea'] > 4000) &
            #         (df[cls.target_col] < 300000)].index,
            #         inplace=True)
            pass
        return df

    def fill_by_type(cls, x, col):
        if pd.isna(x):
            if col.dtype == 'object':
                return 0
            return 0
        return x

    def fill_na(cls, df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: cls.fill_by_type(x, df[col]))
        return df

    def get_encoding_lookup(cls, cols):
        df = cls.get_df('train')
        target = cls.target_col
        suffix = '_E'
        result = pd.DataFrame()
        for cat_feat in cols:
            cat_feat_target = df[[cat_feat, target]].groupby(cat_feat)
            cat_feat_encoded_name = cat_feat + suffix
            order = pd.DataFrame()
            order['val'] = df[cat_feat].unique()
            order.index = order.val
            order.drop(columns=['val'], inplace=True)
            order[target + '_mean'] = cat_feat_target[[target]].median()
            order['feature'] = cat_feat
            order['encoded_name'] = cat_feat_encoded_name
            order = order.sort_values(target + '_mean')
            order['num_val'] = range(1, len(order)+1)
            result = result.append(order)
        result.reset_index(inplace=True)
        return result

    def get_scaled_categorical(cls, encoding_lookup):
        scaled = encoding_lookup.copy()
        target = cls.target_col
        for feature in scaled['feature'].unique():
            values = scaled[scaled['feature'] == feature]['num_val'].values
            medians = scaled[
                    scaled['feature'] == feature][target + '_mean'].values
            for median in medians:
                scaled_value = ((values.min() + 1) *
                                (median / medians.min()))-1
                scaled.loc[(scaled['feature'] == feature) &
                           (scaled[target + '_mean'] == median),
                           'num_val'] = scaled_value
        return scaled

    def encode_with_lookup(cls, df, encoding_lookup):
        for encoded_index, encoded_row in encoding_lookup.iterrows():
            feature = encoded_row['feature']
            encoded_name = encoded_row['encoded_name']
            value = encoded_row['val']
            encoded_value = encoded_row['num_val']
            df.loc[df[feature] == value, encoded_name] = encoded_value
        return df

    def encode_onehot(cls, df, cols):
        df = pd.concat([df, pd.get_dummies(df[cols], drop_first=True)], axis=1)
        return df

    def encode_categorical(cls, df, cols=[], method='one_hot'):
        if len(cols) == 0:
            cols = cls.get_categorical().columns.values
        if method == 'target_mean':
            encoding_lookup = cls.get_encoding_lookup(cols)
            encoding_lookup = cls.get_scaled_categorical(encoding_lookup)
            df = cls.encode_with_lookup(df, encoding_lookup)
        if method == 'one_hot':
            if len(set(cols) - set(cls.get_dtype(include_type=['object'])
                   .columns.values)) > 0:
                    for col in cols:
                        df[col] = df[col].apply(lambda x: str(x))
            df = cls.encode_onehot(df, cols)
        df.drop(cols, axis=1, inplace=True)
        return df

    def fix_zero_infinity(cls, x):
        if (x == 0) or math.isinf(x):
            return 0
        return x

    def normalize_features(cls, df, cols=[]):
        if len(cols) == 0:
            cols = cls.get_numeric().columns.values
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x:
                                        np.log1p(x).astype('float64'))
                df[col] = df[col].apply(lambda x: cls.fix_zero_infinity(x))
        return df

    def scale_quant_features(cls, df, cols):
        scaler = StandardScaler()
        scaler.fit(df[cols])
        scaled = scaler.transform(df[cols])
        for i, col in enumerate(cols):
            df[col] = scaled[:, i]
        return df

    def drop_ignore(cls, df):
        for col in cls.ignore:
            try:
                df.drop(col, axis=1, inplace=True)
            except Exception:
                pass
        return df

    def drop_low_corr(cls, df, threshold=0.12):
        to_drop = pd.DataFrame(columns=['drop'])
        corr_mat = cls.get_correlations()
        target = cls.target_col
        to_drop['drop'] = corr_mat[(abs(corr_mat[target]) <= threshold)].index
        df.drop(to_drop['drop'], axis=1, inplace=True)
        return df


class Engineer:

    def matches_df(cls, df, images_df):
        return len(set(images_df['PetID'].unique()) -
                   set(df['PetID'].unique())) == 0

    def create_image_df(cls, df, images_df_file_name):
        print('Building new image data csv...', df.name)
        json_folder_path = path + '/input/' + df.name + '_metadata/'
        json_files = [f_name for f_name in os.listdir(json_folder_path)
                      if f_name.endswith('.json')]
        pet_type_dict = {1: 'dog', 2: 'cat'}
        all_images_list = []
        for index, f_name in enumerate(json_files):
            with open(os.path.join(json_folder_path, f_name)) as json_file:
                json_text = json.load(json_file)
                try:
                    label_annotations = json_text['labelAnnotations']
                except Exception:
                    continue
                image_data = pd.DataFrame(label_annotations)\
                    .drop(['mid', 'topicality'], axis=1)\
                    .rename({'description': 'Description',
                            'score': 'Score'},
                            axis=1)
                pet_id = f_name.split('-')[0]
                image_data['PetID'] = pet_id
                image_data['ImageID'] = int(f_name.split('-')[1].split('.')[0])
                image_data['PetLabel'] = pet_type_dict[
                    df[df['PetID'] == pet_id]['Type'].values[0]]
                if df.name == 'train':
                    image_data['AdoptionSpeed'] = \
                        df[df['PetID'] == pet_id]['AdoptionSpeed']
            all_images_list.append(image_data)
        images_df = pd.concat(all_images_list)
        images_df.to_csv(images_df_file_name, index=False)
        return images_df

    def get_image_data(cls, df, force_csv=False):
        images_df_file_name = path + '/' + df.name + '_image_data.csv'
        try:
            images_df = pd.read_csv(images_df_file_name)
            no_file = False
        except Exception:
            no_file = True
        if no_file or force_csv or not cls.matches_df(df, images_df):
            images_df = cls.create_image_df(df, images_df_file_name)
        return images_df

    def rate_image(cls, x):
        pet_label = x['PetLabel']
        score = x['Score']
        if pet_label == 'dog':
            good_threshold = 0.96
        if pet_label == 'cat':
            good_threshold = 0.99
        if score > good_threshold:
            return 2
        return 1

    def cap_max_image_rating(cls, x):
        if x > 2:
            return 2
        return x

    def append_image_data(cls, df):
        images_df = cls.get_image_data(df)
        images = images_df[(images_df['PetLabel'] ==
                            images_df['Description'])][
                            ['PetID', 'Score', 'PetLabel', 'ImageID']]
        images['ImageRating'] = images[['PetLabel', 'Score']]\
            .apply(lambda x: cls.rate_image(x), axis=1)
        first = images[images['ImageID'] == 1][['PetID', 'ImageRating']]
        first.rename({'ImageRating': 'FirstImageRating'},
                     axis=1, inplace=True)
        seconds = images[(images['ImageID'] > 1) &
                         (images['ImageRating'] > 1)]\
            .groupby('PetID')['ImageRating'].count().reset_index()
        seconds.rename({'ImageRating': 'SecondImageRating'},
                       axis=1, inplace=True)
        seconds['SecondImageRating'] = seconds['SecondImageRating']\
            .apply(lambda x: cls.cap_max_image_rating(x))
        image_ratings = pd.merge(first, seconds, on='PetID', how='left')
        df = pd.merge(df, image_ratings[['PetID', 'FirstImageRating',
                                        'SecondImageRating']],
                      on='PetID', how='left')
        df['FirstImageRating'].fillna(0, inplace=True)
        df['SecondImageRating'].fillna(0, inplace=True)
        df['TotalImageRating'] = df['FirstImageRating'] +\
            (df['SecondImageRating'] * .1)
        df.loc[df['TotalImageRating'] == 1.0, 'TotalImageRating'] = 0.0
        df.loc[(df['TotalImageRating'] == 1.2) |
               (df['TotalImageRating'] == 1.1), 'TotalImageRating'] = 1.0
        df.loc[df['TotalImageRating'] == 2.1, 'TotalImageRating'] = 2.0
        df.loc[df['TotalImageRating'] == 2.2, 'TotalImageRating'] = 3.0
        return df

    def get_top_rescuers(cls, x, top_rescuers):
        if x in top_rescuers:
            return x
        return False

    def rescuer(cls, df):
        top_rescuers = list(df['RescuerID'].value_counts().index[:5])
        df['Big_Rescuer'] = df['RescuerID']\
            .apply(lambda x: cls.get_top_rescuers(x, top_rescuers))
        return df

    def fee(cls, df):
        df.loc[df['Fee'] > 0, 'Has_Fee'] = True
        df.loc[df['Fee'] == 0, 'Has_Fee'] = False
        return df

    def photo(cls, df):
        df.loc[df['PhotoAmt'] > 1, 'Has_2Photos'] = True
        df.loc[df['PhotoAmt'] < 2, 'Has_2Photos'] = False
        # df.loc[df['VideoAmt'] > 0, 'Has_Video'] = True
        # df.loc[df['VideoAmt'] == 0, 'Has_Video'] = False
        return df

    def simplify_name_length(cls, x):
        length = len(str(x))
        if length < 3:
            return 'short'
        # if length < 20:
        #     return 'medium'
        # if length > 19:
        #     return 'long'
        return 'long'

    def name_length(cls, df):
        df['NameLength'] = df['Name']\
            .apply(lambda x: cls.simplify_name_length(x))
        return df

    def get_name_groups(cls, df):
        names = {}
        names_by_count = df[df['Type'] == 1]['Name']\
            .value_counts().index.tolist()
        top5 = [a.lower() for a in names_by_count[:5]]
        top30 = [a.lower() for a in names_by_count[:30]]
        rest = [a.lower() for a in names_by_count[:]]
        names['dog'] = {
            'top5': top5,
            'top30': top30,
            'rest': rest
        }
        names_by_count = df[df['Type'] == 2]['Name']\
            .value_counts().index.tolist()
        top5 = [a.lower() for a in names_by_count[:5]]
        top30 = [a.lower() for a in names_by_count[:30]]
        rest = [a.lower() for a in names_by_count[:]]
        names['cat'] = {
            'top5': top5,
            'top30': top30,
            'rest': rest
        }
        return names

    def simplify_names(cls, x, names):
        x = str(x)
        x = x.lower()
        if 'nan' in x:
            return 'NAN'
        if x in names['top5']:
            return 'top5'
        # if x in names['top30']:
        #     return 'top30'
        # if '&' in x:
        #     return 'and'
        if x in names['rest']:
            return 'rest'

    def names(cls, df):
        names = cls.get_name_groups(df)
        df.loc[df['Type'] == 1, 'NameGroup'] = df[df['Type'] == 1]['Name']\
            .apply(lambda x: cls.simplify_names(x, names['dog']))
        df.loc[df['Type'] == 2, 'NameGroup'] = df[df['Type'] == 2]['Name']\
            .apply(lambda x: cls.simplify_names(x, names['cat']))
        return df

    def color(cls, df):
        df.loc[(df['Color3'] > 0) | (df['Color2'] > 0),
               'Mixed_Color'] = True
        df.loc[(df['Color3'] == 0) | (df['Color2'] == 0),
               'Mixed_Color'] = False
        return df

    def simplify_quantity(cls, df):
        bins = (0, 1, 10, 100)
        group_names = ['solo', 'litter', 'herd']
        categories = pd.cut(df['Quantity'], bins, labels=group_names)
        return categories

    def quantity(cls, df):
        df.loc[df['Quantity'] == 0, 'Quantity'] = 1
        df.loc[df['Quantity'] > 0, 'Quantity'] = 2
        return df

    def gender(cls, df):
        df.loc[(df['Gender'] == 3) &
               (df['Quantity'] == 2), 'Gender'] = 1.5
        df.loc[(df['Gender'] == 3) &
               (df['Quantity'] > 2), 'Gender'] = 0
        return df

    def breed(cls, df):
        # df.loc[df['Breed2'] > 0, 'Mixed_Breed'] = True
        # df.loc[df['Breed2'] == 0, 'Mixed_Breed'] = False
        df.loc[df['Breed1'] == 307, 'Mixed_Breed'] = True
        df.loc[df['Breed1'] != 307, 'Mixed_Breed'] = False
        return df

    def numerize_features(cls, df, cols):
        train, test = cls.get_dfs()
        df_combined = pd.concat([train[cols], test[cols]])
        train.drop(cls.target_col, axis=1, inplace=True)
        for feature in cols:
            le = LabelEncoder()
            df_combined[feature] = df_combined[feature].apply(lambda x: str(x))
            df[feature] = df[feature].apply(lambda x: str(x))
            le = le.fit(df_combined[feature])
            df[feature] = le.transform(df[feature])
        return df

    def simplify_ages(cls, df, animal):
        if animal == 'dog':
            bins = (-1, 0, 2, 256)
            group_names = ['baby', 'child', 'adult']
            categories = pd.cut(df[df['Type'] == 1]['Age'], bins,
                                labels=group_names)
        if animal == 'cat':
            bins = (-1, 4, 256)
            group_names = ['baby', 'adult']
            categories = pd.cut(df[df['Type'] == 2]['Age'], bins,
                                labels=group_names)
        return categories

    def age(cls, df):
        df.loc[df['Type'] == 1, 'AgeGroup'] = cls.simplify_ages(df, 'dog')
        df.loc[df['Type'] == 2, 'AgeGroup'] = cls.simplify_ages(df, 'cat')
        df.drop('Age', axis=1, inplace=True)
        return df

    def sum_features(cls, df, col_sum):
        for col_set in col_sum:
            f_name = '__'.join(col_set[:])
            df[f_name] = df[[*col_set]].sum(axis=1)
            df.drop(col_set, axis=1, inplace=True)
        return df

    def combine_features(cls, row, col_set):
        result = ''
        for col in col_set:
            if result != '':
                result += '_'
            result += str(row[col])
        return result

    def combine(cls, df, col_sets):
        for col_set in col_sets:
            f_name = '__'.join(col_set[:])
            df[f_name] = df.apply(lambda x: cls.combine_features(x, col_set),
                                  axis=1)
            df.drop(col_set, axis=1, inplace=True)
        return df

    def multiply_features(cls, df, feature_sets):
        for feature_set in feature_sets:
            # multipled_name = '_x_'.join(feature_set[:])
            # df.drop(feature_set, axis=1, inplace=True)
            pass
        return df


class Model:

    def forward_selection(cls, df, features_count=1):
        if df.name == 'train':
            qwk_scorer = make_scorer(cls.quadratic_weighted_kappa,
                                     greater_is_better=True)
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            X = df.drop('AdoptionSpeed', axis=1)
            y = df['AdoptionSpeed']
            X_train, X_test,\
                y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                   random_state=42)
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            sfs1 = sfs(model,
                       k_features=3,
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring=qwk_scorer,
                       cv=5)
            sfs1 = sfs1.fit(X_train, y_train)
            best_cols = list(sfs1.k_feature_idx_)
        return best_cols

    def confusion_matrix(cls, rater_a, rater_b,
                         min_rating=None, max_rating=None):
        """
        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
        Returns the confusion matrix between rater's ratings
        """
        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(rater_a + rater_b)
        if max_rating is None:
            max_rating = max(rater_a + rater_b)
        num_ratings = int(max_rating - min_rating + 1)
        conf_mat = [[0 for i in range(num_ratings)]
                    for j in range(num_ratings)]
        for a, b in zip(rater_a, rater_b):
            conf_mat[a - min_rating][b - min_rating] += 1
        return conf_mat

    def histogram(cls, ratings, min_rating=None, max_rating=None):
        """
        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
        Returns the counts of each type of rating that a rater made
        """
        if min_rating is None:
            min_rating = min(ratings)
        if max_rating is None:
            max_rating = max(ratings)
        num_ratings = int(max_rating - min_rating + 1)
        hist_ratings = [0 for x in range(num_ratings)]
        for r in ratings:
            hist_ratings[r - min_rating] += 1
        return hist_ratings

    def quadratic_weighted_kappa(cls, rater_a, rater_b,
                                 min_rating=0, max_rating=4):
        """
        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
        Calculates the quadratic weighted kappa
        quadratic_weighted_kappa calculates the quadratic weighted kappa
        value, which is a measure of inter-rater agreement between two raters
        that provide discrete numeric ratings.  Potential values range from -1
        (representing complete disagreement) to 1 (representing complete
        agreement).  A kappa value of 0 is expected if all agreement is due to
        chance.
        quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
        each correspond to a list of integer ratings.  These lists must have the
        same length.
        The ratings should be integers, and it is assumed that they contain
        the complete range of possible ratings.
        quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
        is the minimum possible rating, and max_rating is the maximum possible
        rating
        """
        rater_a = np.array(rater_a, dtype=int)
        rater_b = np.array(rater_b, dtype=int)
        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(min(rater_a), min(rater_b))
        if max_rating is None:
            max_rating = max(max(rater_a), max(rater_b))
        conf_mat = cls.confusion_matrix(rater_a, rater_b,
                                    min_rating, max_rating)
        num_ratings = len(conf_mat)
        num_scored_items = float(len(rater_a))

        hist_rater_a = cls.histogram(rater_a, min_rating, max_rating)
        hist_rater_b = cls.histogram(rater_b, min_rating, max_rating)

        numerator = 0.0
        denominator = 0.0

        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                  / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items

        return 1.0 - numerator / denominator

    def fix_shape(cls, df):
        df_name = df.name
        if df_name == 'train':
            cols_to_add = set(cls.get_df('test').columns.values) -\
                          set(df.drop(cls.target_col, axis=1).columns.values)
        if df_name == 'test':
            cols_to_add = set(cls.get_df('train').drop(cls.target_col, axis=1)
                              .columns.values) - set(df.columns.values)
        cols_to_add = np.array(list(cols_to_add))
        cols_to_add = np.append(cols_to_add, df.columns.values)
        df = df.reindex(columns=cols_to_add, fill_value=0)
        df.name = df_name
        return df

    def cross_validate(cls, model, parameters):
        train, test = cls.get_dfs()
        # TODO: check if there are lists in parameters to run gridsearch
        if len(train.drop(cls.target_col,
               axis=1).columns) != len(test.columns):
            cls.mutate(cls.fix_shape)
            train = cls.get_df('train')
        scores = np.array([])
        skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        X = train.drop(columns=[cls.target_col])
        y = train[cls.target_col]
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            cv_model = model(**parameters)
            cv_model.fit(X_train, y_train)
            X_predictions = cv_model.predict(X_test)
            score = cls.quadratic_weighted_kappa(y_test, X_predictions, 0, 4)
            scores = np.append(scores, score)
        score = np.round(scores.mean(), decimals=5)
        return score

    def fit(cls, model, parameters):
        train = cls.get_df('train')
        X = train.drop(columns=[cls.target_col])
        y = train[cls.target_col]
        model = model(**parameters)
        model.fit(X, y)
        importances = model.feature_importances_
        for col, importance in zip(X.columns.values, importances):
            print(col, importance)
        return model

    def predict(cls, model):
        test = cls.get_df('test')
        predictions = model.predict(test)
        return predictions

    def save_predictions(cls, predictions, score=0, id_col=False):
        now = str(time.time()).split('.')[0]
        df = cls.get_df('test', False, True)
        target = cls.target_col
        if not id_col:
            id_col = df.columns[0]
        df[target] = predictions
        if not os.path.exists(path + '/output'):
            os.makedirs(path + '/output')
        if os.path.exists(path + '/output'):
            df[[id_col,
                target]].to_csv(path + '/output/submit__'
                                + str(int(score * 100000))
                                + '__' + now +
                                '.csv', index=False)
        df[[id_col, target]].to_csv('submission.csv', index=False)


class Data(Explore, Clean, Engineer, Model):

    def __init__(self, train_csv, test_csv, target='',
                 ignore=[], keep=[], col_sum=[]):
        '''Create pandas DataFrame objects for train and test data.

        Positional arguments:
        train_csv -- relative path to training data in csv format.
        test_csv -- relative path to test data in csv format.

        Keyword arguments:
        target -- target feature column name in training data.
        ignore -- columns names in list to ignore during analyses.
        '''
        self.__train = pd.read_csv(train_csv)
        self.__test = pd.read_csv(test_csv)
        self.__train.name, self.__test.name = self.get_df_names()
        self.target_col = target
        self.ignore = ignore
        self.keep = keep
        self.col_sum = col_sum
        self.__original = False
        self.__log = False
        self.check_in()
        self.debug = False

    def __str__(cls):
        train_columns = 'Train: \n"' + '", "'.join(cls.__train.head(2)) + '"\n'
        test_columns = 'Test: \n"' + '", "'.join(cls.__test.head(2)) + '"\n'
        return train_columns + test_columns

    def get_df_names(cls):
        return ('train', 'test')

    def get_dfs(cls, ignore=False, originals=False, keep=False):
        train, test = (cls.__train.copy(),
                       cls.__test.copy())
        if originals:
            train, test = (cls.__original)
        if ignore:
            train, test = (train.drop(columns=cls.ignore),
                           test.drop(columns=cls.ignore))
        if keep:
            train, test = (train[cls.keep],
                           test[cls.keep])
        train.name, test.name = cls.get_df_names()
        return (train, test)

    def get_df(cls, name, ignore=False, original=False, keep=False):
        train, test = cls.get_dfs(ignore, original, keep)
        if name == 'train':
            return train
        if name == 'test':
            return test

    def log(cls, entry=False, status=False):
        if cls.__log is False:
            cls.__log = pd.DataFrame(columns=['entry', 'status'])
        log_entry = pd.DataFrame({'entry': entry, 'status': status}, index=[0])
        cls.__log = cls.__log.append(log_entry, ignore_index=True)
        if status == 'Fail':
            cls.rollback()
        else:
            cls.check_out()
            if cls.debug:
                cls.print_log()

    def print_log(cls):
        print(cls.__log)

    def check_in(cls):
        cls.__current = cls.get_dfs()
        if cls.__original is False:
            cls.__original = cls.__current

    def check_out(cls):
        cls.__previous = cls.__current
        cls.__train.name, cls.__test.name = cls.get_df_names()

    def rollback(cls):
        try:
            cls.__train, cls.__test = cls.__previous
            status = 'Success - To Previous'
        except Exception:
            cls.__train, cls.__test = cls.__original
            status = 'Success - To Original'
        cls.log('rollback', status)

    def reset(cls):
        cls.__train, cls.__test = cls.__original
        cls.log('reset', 'Success')

    def update_dfs(cls, train, test):
        train.name, test.name = cls.get_df_names()
        cls.__train = train
        cls.__test = test

    def mutate(cls, mutation, *args):
        '''Make changes to both train and test DataFrames.
        Positional arguments:
        mutation -- function to pass both train and test DataFrames to.
        *args -- arguments to pass to the function, following each DataFrame.

        Example usage:
        def multiply_column_values(df, col_name, times=10):
            #do magic...

        Data.mutate(multiply_column_values, 'Id', 2)
        '''
        cls.check_in()
        try:
            train = mutation(cls.get_df('train'), *args)
            test = mutation(cls.get_df('test'), *args)
            cls.update_dfs(train, test)
            status = 'Success'
        except Exception:
            print(traceback.print_exc())
            status = 'Fail'
        cls.log(mutation.__name__, status)


def run(d, model, parameters):
    mutate = d.mutate
    # mutate(d.sample, [[0, 1]])
    # print(d.get_df('train')['AdoptionSpeed'].value_counts())
    mutate(d.append_image_data)
    mutate(d.sample_ros)
    # mutate(d.rescuer)
    # mutate(d.age)
    # mutate(d.gender)
    # mutate(d.quantity)
    # mutate(d.names)
    # mutate(d.name_length)
    # mutate(d.color)
    # mutate(d.breed)
    # mutate(d.fee)
    # mutate(d.photo)
    # mutate(d.sum_features, d.col_sum)
    # mutate(d.combine, [
    #     # ['Breed1', 'Breed2'],
    #     ['Color1', 'Color2']
    #     ])
    # mutate(d.fill_na)
    # mutate(d.numerize_features, [
    # #     #    'Breed1',
    #     #    'Color1__Color2'
    #         # 'Type'
    #        ])
    # mutate(d.encode_categorical, [
    #        'AgeGroup',
    #     #    'NameLength',
    #     #    'Is_Solo',
    #     #    'Has_2Photos',
    #        ])
    mutate(d.drop_ignore)
    print(d.get_df('train').columns)
    # best_features = d.forward_selection(d.get_df('train'), 5)
    # print('Best Features', best_features)
    # sys.exit()
    score = d.cross_validate(model, parameters)
    print('Score: ', score)
    print(d.get_df('train').head(2))
    model = d.fit(model, parameters)
    predictions = d.predict(model)
    d.print_log()
    return (predictions, score)


path = '.'
if os.getcwd().split('/')[1] == 'kaggle':
    path = '..'

zip_files = list(filter(lambda x: '.zip' in x, os.listdir(path + '/input/')))


def unzip(file):
    to_unzip = path + '/input/' + file
    destination = path + '/input/' + file.split('.')[0]
    with zipfile.ZipFile(to_unzip, 'r') as zip_ref:
        zip_ref.extractall(destination)


def move_zips(move_from, move_to):
    zip_files = list(filter(lambda x: '.zip' in x, os.listdir(move_from)))
    if not os.path.exists(move_to):
        os.makedirs(move_to)
    for file in zip_files:
        shutil.move(move_from + file, move_to + file)


if len(zip_files) > 0:
    for file in zip_files:
        unzip(file)
    move_zips(path + '/input/', path + '/input/source_zips/')

model = RandomForestClassifier
parameters = {
    'n_estimators': 100,
    'min_samples_split': 50
}
cols_to_ignore = ['PetID',
                  'RescuerID',
                  'Description',
                  'Name',
                  'Type',
                #   'Age',
                #   'Breed1',
                #   'Breed2',
                  'Gender',
                #   'Color1',
                #   'Color2',
                  'Color3',
                  'MaturitySize',
                #   'FurLength',
                  'Vaccinated',
                  'Dewormed',
                  'Sterilized',
                  'Health',
                  'Quantity',
                  'Fee',
                #   'State',
                  'VideoAmt',
                #   'PhotoAmt',
                  # Custom:
                #   'FirstImageRating',
                  'SecondImageRating',
                  'TotalImageRating'
                  ]
id_col = 'PetID'
d = Data(path + '/input/train/train.csv',
         path + '/input/test/test.csv',
         'AdoptionSpeed',
         ignore=cols_to_ignore)
predictions, score = run(d, model, parameters)
d.save_predictions(predictions, score, id_col)
