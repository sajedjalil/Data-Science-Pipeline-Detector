import pandas as pd
import math
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import os

pd.options.display.max_columns = 100

def touch(fname, times=None):
        with open(fname, 'a'):
                    os.utime(fname, times)

def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def add_play_physics(play):
    # Format columns
    play['time'] = pd.to_datetime(play['time'])
    # Distance
    play['dis_meters'] = play['dis'] / 1.0936  # Add distance in meters
    # Speed
    play['dis_meters'] / 0.01
    play['v_mps'] = play['dis_meters'] / 0.1
    # Angles to radians
    play['dir_radians'] = play['dir'].apply(math.radians)
    play['o_radians'] = play['o'].apply(math.radians)
    average_weight_nfl_pounds = 245.86
    average_weight_nfl_kg = average_weight_nfl_pounds * 0.45359237
    # http://webpages.uidaho.edu/~renaes/251/HON/Student%20PPTs/Avg%20NFL%20ht%20wt.pdf
    play['momentum'] = play['v_mps'] * average_weight_nfl_kg
    play['momentum_x'] = pol2cart(play['momentum'], play['dir_radians'])[0]
    play['momentum_y'] = pol2cart(play['momentum'], play['dir_radians'])[1]
    return play


"""
This code loops through every play and:
    1. For each moment in time of the play, for each player in the play:
        - Finds the closest other player to them.
        - Computes the resulting force of the two in relation to eachother.
            - If the force is higher this indicates a higher danger probability.
"""


def calculate_risk(play):
    """
    Calculate the momentum risk
    """
    play = add_play_physics(play)
    playexpanded = pd.merge(play, play, on=[
                            'season_year', 'gamekey', 'playid', 'time'], suffixes=('', '_partner'))
    # Drop when equal to self
    playexpanded = playexpanded.loc[playexpanded['gsisid']
                                    != playexpanded['gsisid_partner']]
    # Calculate distance to partner
    playexpanded['dist'] = np.sqrt((playexpanded['x'] - playexpanded['x_partner']).apply(
        np.square) + (playexpanded['y'] - playexpanded['y_partner']).apply(np.square))

    playexpanded['opp_momentum'] = np.sqrt(np.square(
        playexpanded['momentum_x'] - playexpanded['momentum_x_partner']) +
        np.square(playexpanded['momentum_y'] - playexpanded['momentum_y_partner']))
    playexpanded['risk_factor'] = playexpanded['opp_momentum'] / \
        playexpanded['dist']
    return playexpanded


pi = pd.read_csv('../input/play_information.csv')

start = timer()
year = 2016
gamekey = 332
playid = 2983

print('Running for {} {} {}'.format(year, gamekey, playid))
plays = pd.read_csv('../input/NGS-2016-post.csv')
plays.columns = [col.lower() for col in plays.columns]

play = plays.loc[(plays['season_year'] == year) &
                 (plays['gamekey'] == gamekey) &
                 (plays['playid'] == playid)]

print(play.dtypes)

play_processed = calculate_risk(play)

play_processed.to_parquet('{}-{}-{}-risk.parquet'.format(year, gamekey, playid))
play_processed.to_csv('{}-{}-{}-risk.csv'.format(year, gamekey, playid))