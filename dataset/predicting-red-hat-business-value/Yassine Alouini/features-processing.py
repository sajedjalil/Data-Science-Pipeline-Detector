
# In this script, I use the Pandas modern approach (with piping) to clean and
# extract features.
# To read more about modern Pandas, check these (awesome) blog posts:
# https://tomaugspurger.github.io/modern-1.html
# Some duplicate code is still there.
# Any (constructive) suggestion to improve it is more than welcome!
# Notice that the feature extraction and cleaning ideas are mostly inspired
# from the work of ZFTurbo. You can find it here:
# https://www.kaggle.com/zfturbo/predicting-red-hat-business-value/xredboost/log


import numpy as np
import pandas as pd

# --------------------------------------------------------#

# Loading the different datasets

act_train_df = pd.read_csv("../input/act_train.csv",
                           dtype={'people_id': np.str,
                                  'activity_id': np.str,
                                  'outcome': np.int8},
                           parse_dates=['date'])


act_test_df = pd.read_csv("../input/act_test.csv",
                          dtype={'people_id': np.str,
                                 'activity_id': np.str},
                          parse_dates=['date'])

people_df = pd.read_csv("../input/people.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               # This is the only numerical
                               #column in char_* list
                               'char_38': np.int32},
                        parse_dates=['date'])

# --------------------------------------------------------#

# Helper functions


def intersect(l_1, l_2):
    return list(set(l_1) & set(l_2))


def get_features(train, test):
    intersecting_features = intersect(train.columns, test.columns)
    intersecting_features.remove('people_id')
    intersecting_features.remove('activity_id')
    return sorted(intersecting_features)

# --------------------------------------------------------#

# Processing functions


def process_date(input_df):
    df = input_df.copy()
    return (df.assign(year=lambda df: df.date.dt.year,  # Extract year
                      month=lambda df: df.date.dt.month,  # Extract month
                      day=lambda df: df.date.dt.day)  # Extract day
            .drop('date', axis=1))


def process_activity_category(input_df):
    df = input_df.copy()
    return df.assign(activity_category=lambda df:
                     df.activity_category.str.lstrip('type ').astype(np.int32))


def process_activities_char(input_df, columns_range):
    """
    Extract the integer value from the different char_* columns in the
    activities dataframes. Fill the missing values with -999 as well
    """
    df = input_df.copy()
    char_columns = ['char_' + str(i) for i in columns_range]
    return (df[char_columns].fillna('type -999')
            .apply(lambda col: col.str.lstrip('type ').astype(np.int32))
            .join(df.drop(char_columns, axis=1)))

# TODO: Extract the magic range (1 to 11) programmatically


def activities_processing(input_df):
    """
    This function combines the date, activity_category and char_*
    columns transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(process_activity_category)
              .pipe(process_activities_char, range(1, 11)))


def process_group_1(input_df):
    df = input_df.copy()
    return df.assign(group_1=lambda df:
                     df.group_1.str.lstrip('group ').astype(np.int32))


# TODO: Refactor the different *_char functions

def process_people_cat_char(input_df, columns_range):
    """
    Extract the integer value from the different categorical char_*
    columns in the people dataframe.
    """
    df = input_df.copy()
    cat_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[cat_char_columns].apply(lambda col:
                                       col.str.lstrip('type ').astype(np.int32))
                                .join(df.drop(cat_char_columns, axis=1)))


def process_people_bool_char(input_df, columns_range):
    """
    Extract the integer value from the different boolean char_* columns in the
    people dataframe.
    """
    df = input_df.copy()
    boolean_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[boolean_char_columns].apply(lambda col: col.astype(np.int32))
                                    .join(df.drop(boolean_char_columns,
                                                  axis=1)))


# TODO: Extract the magic ranges (1 to 10 and 10 to 38) programmatically

def people_processing(input_df):
    """
    This function combines the date, group_1 and char_*
    columns (inclunding boolean and categorical ones) transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(process_group_1)
              .pipe(process_people_cat_char, range(1, 10))
              .pipe(process_people_bool_char, range(10, 38)))


def merge_with_people(input_df, people_df):
    """
    Merge (left) the given input dataframe with the people dataframe and
    fill the missing values with -999.
    """
    df = input_df.copy()
    return (df.merge(people_df, how='left', on='people_id',
                     left_index=True, suffixes=('_activities', '_people'))
            .fillna(-999))

# --------------------------------------------------------#

# Processing pipelines

processed_people_df = people_df.pipe(people_processing)
train_df = (act_train_df.pipe(activities_processing)
                        .pipe(merge_with_people, processed_people_df))
test_df = (act_test_df.pipe(activities_processing)
                      .pipe(merge_with_people, processed_people_df))

# --------------------------------------------------------#

# Output

features_list = get_features(train_df, test_df)

print("The merged features are: ")
print('\n'.join(features_list), "\n")
print("The train dataframe head is", "\n")
print(train_df.head())
print("The test dataframe head is", "\n")
print(test_df.head())
