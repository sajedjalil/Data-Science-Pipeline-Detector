import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import time

def main():

    start = time.time()
    print("start")
    data = pd.read_csv('../data.csv')

    data.drop('matchup', axis=1, inplace=True)
    data.drop('team_id', axis=1, inplace=True)
    data.drop('team_name', axis=1, inplace=True)
    data.drop('game_id', axis=1, inplace=True)
    data.drop('game_date', axis=1, inplace=True)
    data.drop('game_event_id', axis=1, inplace=True)
# 	'''
# 	# Engineer game date by secotion of the season
# 	data['month_of_season'] = [i[5:7] for i in data['game_date']]
# 	date_dict = {'1':'4/7', '2':'5/7', '3':'6/7', '4':'7/7', '10':'1/7', '11':'2/7', '12':'3/7'}
# 	data['season_interval'] = data['month_of_season'].map(date_dict)
# 	data.drop(['month_of_season', 'game_date'], axis=1, inplace=True)

# 	# Dummy newly created season_interval feature
# 	season_interval = pd.get_dummies(data['season_interval'], prefix='dum')
# 	data = pd.concat((data, season_interval), axis=1)
# 	data.drop('season_interval', axis=1, inplace=True)
	
# 	# Possibly hand-craft dummies for game_event_id
# 	game_event_id = pd.get_dummies(data['game_event_id'], prefix='dum')
# 	data = pd.concat((data, game_event_id), axis=1)
# 	data.drop('game_event_id', axis=1, inplace=True)
# 	'''
	# Dummies for opponent
    opponent = pd.get_dummies(data['opponent'], prefix='dum')
    data = pd.concat((data, opponent), axis=1)
    data.drop('opponent', axis=1, inplace=True)

	# Dummies for action_type
    action_type = pd.get_dummies(data['action_type'], prefix='dum')
    data = pd.concat((data, action_type), axis=1)
    data.drop('action_type', axis=1, inplace=True)
# 	'''
# 	# Degree of difficulty in action_type [0 (easy) - 5 (difficult)]
# 	degree_of_difficulty = { 

# 		 'Alley Oop Dunk Shot':2,
# 		 'Alley Oop Layup shot':2,
# 		 'Cutting Finger Roll Layup Shot':2,
# 		 'Cutting Layup Shot':2,
# 		 'Driving Bank shot':4,
# 		 'Driving Dunk Shot':2,
# 		 'Driving Finger Roll Layup Shot':2,
# 		 'Driving Finger Roll Shot':2,
# 		 'Driving Floating Bank Jump Shot':4,
# 		 'Driving Floating Jump Shot':4, 
# 		 'Driving Hook Shot':3, 
# 		 'Driving Jump shot':4,
# 		 'Driving Layup Shot':2, 
# 		 'Driving Reverse Layup Shot':4, 
# 		 'Driving Slam Dunk Shot':2,
# 		 'Dunk Shot':1, 
# 		 'Fadeaway Bank shot':5, 
# 		 'Fadeaway Jump Shot':5,
# 		 'Finger Roll Layup Shot':1,
# 		 'Finger Roll Shot':2, 
# 		 'Floating Jump shot':4,
# 		 'Follow Up Dunk Shot':2, 
# 		 'Hook Bank Shot':3, 
# 		 'Hook Shot':3, 
# 		 'Jump Bank Shot':4,
# 		 'Jump Hook Shot':3, 
# 		 'Jump Shot':4, 
# 		 'Layup Shot':2, 
# 		 'Pullup Bank shot':4,
# 		 'Pullup Jump shot':4, 
# 		 'Putback Dunk Shot':2, 
# 		 'Putback Layup Shot':2,
# 		 'Putback Slam Dunk Shot':2,
# 		 'Reverse Dunk Shot':3,
# 		 'Reverse Layup Shot':3,
# 		 'Reverse Slam Dunk Shot':3, 
# 		 'Running Bank shot':4, 
# 		 'Running Dunk Shot':2,
# 		 'Running Finger Roll Layup Shot':2, 
# 		 'Running Finger Roll Shot':2,
# 		 'Running Hook Shot':3, 
# 		 'Running Jump Shot':4, 
# 		 'Running Layup Shot':2,
# 		 'Running Pull-Up Jump Shot':4, 
# 		 'Running Reverse Layup Shot':3,
# 		 'Running Slam Dunk Shot':2, 
# 		 'Running Tip Shot':3, 
# 		 'Slam Dunk Shot':2,
# 		 'Step Back Jump shot':5, 
# 		 'Tip Layup Shot':2,
# 		 'Tip Shot':2, 
# 		 'Turnaround Bank shot':5,
# 		 'Turnaround Fadeaway Bank Jump Shot':5, 
# 		 'Turnaround Fadeaway shot':5,
# 		 'Turnaround Finger Roll Shot':5, 
# 		 'Turnaround Hook Shot':5,
# 		 'Turnaround Jump Shot':5,
# 	}

# 	data['degree_of_difficulty'] = data['action_type'].map(degree_of_difficulty)
	
# 	# Dummy degree of difficulty
# 	degree_of_difficulty = pd.get_dummies(data['degree_of_difficulty'], prefix='dum')
# 	data = pd.concat((data, degree_of_difficulty), axis=1)
# 	data.drop(['action_type', 'degree_of_difficulty'], axis=1, inplace=True)
# 	'''
	# Dummies for combined_shot_type
    combined_shot_type = pd.get_dummies(data['combined_shot_type'], prefix='dum')
    data = pd.concat((data, combined_shot_type), axis=1)
    data.drop('combined_shot_type', axis=1, inplace=True)

	# Dummies for shot_zone_basic
    shot_zone_basic = pd.get_dummies(data['shot_zone_basic'], prefix='True')
    data = pd.concat((data, shot_zone_basic), axis=1)
    data.drop('shot_zone_basic', axis=1, inplace=True)

	# Dummies for shot_zone_range
    shot_zone_range = pd.get_dummies(data['shot_zone_range'], prefix='dum')
    data = pd.concat((data, shot_zone_range), axis=1)
    data.drop('shot_zone_range', axis=1, inplace=True)

	# Dummies for shot_type
    shot_type = pd.get_dummies(data['shot_type'], prefix='dum')
    data = pd.concat((data, shot_type), axis=1)
    data.drop('shot_type', axis=1, inplace=True)

	# Dummies for shot_zone_area
    shot_zone_area = pd.get_dummies(data['shot_zone_area'], prefix='dum')
    data = pd.concat((data, shot_zone_area), axis=1)
    data.drop('shot_zone_area', axis=1, inplace=True)
	
	# Get season year & month
    season_year = [data['season'][i][:4] for i in range(len(data['season']))]
    data['season_year'] = season_year
    season_month = [data['season'][i][-2:] for i in range(len(data['season']))]
    data['season_month'] = season_month

	# Dummies for season year
    season_year_dum = pd.get_dummies(data['season_year'], prefix='dum')
    data = pd.concat((data, season_year_dum), axis=1)
    data.drop('season_year', axis=1, inplace=True)

	# Dummies for season month
    season_month_dum = pd.get_dummies(data['season_month'], prefix='dum')
    data = pd.concat((data, season_month_dum), axis=1)
    data.drop('season_month', axis=1, inplace=True)

    data.drop('season', axis=1, inplace=True)

	# Engineer features corresponding to alternate coordinate systems
    xy_sc = StandardScaler() 
    new_xy = xy_sc.fit_transform(data[['loc_x', 'loc_y']]) 

    data['X'] = new_xy[:, 0]
    data['Y'] = new_xy[:, 1]

    data['rot30_X'] = (1.732/2) * data['X'] + (1./2) * data['Y'] 
    data['rot30_Y'] = (1.732/2) * data['Y'] - (1./2) * data['X']

    data['rot45_X'] = .707 * data['Y'] + .707 * data['X'] 
    data['rot45_Y'] = .707 * data['Y'] - .707 * data['X']

    data['rot60_X'] = (1.0/2) * data['X'] + (1.732/2) * data['Y'] 
    data['rot60_Y'] = (1.0/2) * data['Y'] - (1.732/2) * data['X']

    data['radial_r'] = np.sqrt( np.power(data['Y'], 2) + np.power(data['X'] ,2))
    data.drop(['X', 'Y'], axis=1, inplace=True)
    print("finished")
	# Save new frame
    data.to_csv('../fset_two.csv', index=False)
	
    print('Finished in %3.2f minutes.' % (np.abs(time.time()-start)/60))

if __name__ == '__main__':
    main()
