from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re

# Base url, and a lambda func to return url for a given year
base_url = 'http://kenpom.com/index.php'
url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2016 else base_url)

# Years on kenpom's site (could also scrape this and 
# set as a list if you want to be more dynamic)
years = range(2002, 2017)

# Create a method that parses a given year and spits out a raw dataframe
def import_raw_year(year):
    """
    Imports raw data from a ken pom year into a dataframe
    """
    f = requests.get(url_year(year))
    soup = BeautifulSoup(f.text)
    table_html = soup.find_all('table', {'id': 'ratings-table'})

    # Weird issue w/ <thead> in the html
    # Prevents us from just using pd.read_html
    # Let's find all the thead contents and just replace/remove them
    # This allows us to easily put the table row data into a dataframe using panda
    thead = table_html[0].find_all('thead')

    table = table_html[0]
    for x in thead:
        table = str(table).replace(str(x), '')

    df = pd.read_html(table)[0]
    df['year'] = year
    return df
    

# Import all the years into a singular dataframe
df = None
for x in years:
    df = pd.concat( (df, import_raw_year(x)), axis=0) \
        if df is not None else import_raw_year(2002)

# Column rename based off of original website
df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'Pyth', 
             'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
             'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank', 
             'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
             'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank', 'Year']
             
# Lambda that returns true if given string is a number and a valid seed number (1-16)
valid_seed = lambda x: True if str(x).replace(' ', '').isdigit() \
                and int(x) > 0 and int(x) <= 16 else False

# Use lambda to parse out seed/team
df['Seed'] = df['Team'].apply(lambda x: x[-2:].replace(' ', '') \
                              if valid_seed(x[-2:]) else np.nan )

df['Team'] = df['Team'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)

# Split W-L column into wins and losses
df['Wins'] = df['W-L'].apply(lambda x: int(re.sub('-.*', '', x)) )
df['Losses'] = df['W-L'].apply(lambda x: int(re.sub('.*-', '', x)) )
df.drop('W-L', inplace=True, axis=1)


# Reorder columns just cause I'm OCD
df=df[[ 'Year', 'Rank', 'Team', 'Conference', 'Wins', 'Losses', 'Seed','Pyth', 
             'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
             'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank', 
             'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
             'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank']]
             
df.head(25)
