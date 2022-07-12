"""
This script fixes value conflicts that exist in the geoNetwork attrbites (e.g. 'city' and 'country' disagreement).
For an explanation of why and how this is done, please refer to my notebook kernel at 
https://www.kaggle.com/mithrillion/fixing-conflicts-in-the-geonetwork-attributes

The basic steps of conflict resolution are:
1. Separate the six geographic attributes into two groups
(city, metro, region) (country, continent, subcontinent) that have high inter-group agreement
2. Fix a few missing value cases that are obvious
3. Use highest combination frequency to determine when the two groups disagree, which value should be
chosen as the "correct" one
"""

import pandas as pd
import re
from os import path

ROOT = '../input/gstore-revenue-data-preprocessing'
OUTPUT = '.'

dat = {}
dat['train'] = pd.read_pickle(path.join(ROOT, 'train.pkl'))
dat['test'] = pd.read_pickle(path.join(ROOT, 'test.pkl'))

geo_colnames = [
    c for c in dat['train'].columns if re.match(r'geoNetwork', c) is not None
]
pure_geo_columns = [c for c in geo_colnames if c != 'geoNetwork.networkDomain']

selected = {}
for subset_name in ['train', 'test']:
    # add 'N/A' as a separate category for easier processing later
    for c in pure_geo_columns:
        dat[subset_name].loc[:, c] = dat[subset_name][c].cat.add_categories(
            'N/A').fillna('N/A')
    # select the geo portion of the DFs
    selected[subset_name] = dat[subset_name].loc[:, pure_geo_columns]

combined_selected = pd.concat(
    [selected['train'], selected['test']], axis=0, ignore_index=True)

country_part = combined_selected.groupby(
    ['geoNetwork.continent', 'geoNetwork.country',
     'geoNetwork.subContinent']).size().reset_index()

print(
    "Verifying that (country, continent, subContinent) always appear in unique triples. The following output should be an empty DataFrame."
)
print(country_part[country_part['geoNetwork.country'].duplicated(keep=False)])

# extract "correct" relationship between country, continent and subcontinent
country_part.iloc[:, :3].to_csv(
    path.join(OUTPUT, 'country_continent.csv'), index=False)

# define a known list of city -> region for NA elimination
replacement_pairs = [('Colombo', 'Western Province'), ('Doha', 'Doha'),
                     ('Guatemala City', 'Guatemala Department'), ('Hanoi',
                                                                  'Hanoi'),
                     ('Minsk', 'Minsk Region'), ('Nairobi', 'Nairobi County'),
                     ('Tbilisi', 'Tbilisi'), ('Casablanca',
                                              'Grand Casablanca')]

# operate on both subsets separately
for subset_name in ['train', 'test']:
    # replace known region = N/A cases
    for c, r in replacement_pairs:
        dat[subset_name].loc[(dat[subset_name]['geoNetwork.city'] == c) &
                             (dat[subset_name]['geoNetwork.region'] == 'N/A'),
                             'geoNetwork.region'] = r

    # find most common country values given city and region
    most_common = dat[subset_name].groupby([
        'geoNetwork.city', 'geoNetwork.region'
    ])['geoNetwork.country'].apply(lambda x: x.mode()).reset_index()

    # change all country values to the most common given (city, region)
    for idx, row in most_common.iterrows():
        dat[subset_name].loc[
            (dat[subset_name]['geoNetwork.city'] == row['geoNetwork.city']) &
            (dat[subset_name]['geoNetwork.region'] == row['geoNetwork.region']
             ) & ((dat[subset_name]['geoNetwork.city'] != 'N/A') | (
                 (dat[subset_name]['geoNetwork.region'] != 'N/A'))),
            'geoNetwork.country'] = row['geoNetwork.country']

    # force country and (continent, subcontinent) to agree
    continent_cats = dat[subset_name]['geoNetwork.continent'].cat.categories
    subcont_cats = dat[subset_name]['geoNetwork.subContinent'].cat.categories

    dat[subset_name].drop(
        [
            'geoNetwork.continent', 'geoNetwork.metro',
            'geoNetwork.subContinent'
        ],
        axis=1,
        inplace=True)

    country_continent = country_part.iloc[:, :3]
    dat[subset_name] = pd.merge(
        dat[subset_name],
        country_continent,
        on='geoNetwork.country',
        how='left')
    dat[subset_name]['geoNetwork.continent'] = dat[subset_name][
        'geoNetwork.continent'].astype('category')
    dat[subset_name]['geoNetwork.continent'].cat.set_categories(continent_cats)
    dat[subset_name]['geoNetwork.subContinent'] = dat[subset_name][
        'geoNetwork.subContinent'].astype('category')
    dat[subset_name]['geoNetwork.subContinent'].cat.set_categories(
        subcont_cats)

    for c in pure_geo_columns:
        try:
            dat[subset_name].loc[:, c].cat.remove_categories(
                'N/A', inplace=True)
        except ValueError:
            pass
        except KeyError:
            pass

    dat[subset_name].to_pickle(
        path.join(OUTPUT, 'manual_geo_fix_{0}.pkl'.format(subset_name)))
