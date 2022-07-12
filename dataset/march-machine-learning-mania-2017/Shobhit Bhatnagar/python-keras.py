import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, Activation
from keras.optimizers import Adam
from keras.models import Sequential

def process_data():
	detailed_results = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
	df_1 = pd.DataFrame()
	df_1[['team1','team2']] = detailed_results[['Wteam','Lteam']].copy()
	df_1['pred'] = 1
	df_2 = pd.DataFrame()
	df_2[['team1','team2']] = detailed_results[['Lteam','Wteam']].copy()
	df_2['pred'] = 0
	df = pd.concat((df_1, df_2), axis=0)
	unique_teams = df.team1.unique()
	team_to_int = {team:i for i, team in enumerate(unique_teams)}
	df.team1 = df.team1.apply(lambda x: team_to_int[x])
	df.team2 = df.team2.apply(lambda x: team_to_int[x])
	train = df.values
	np.random.shuffle(train)
	return train, team_to_int


def create_model(num_teams):
	model = Sequential()
	model.add(Embedding(input_dim=num_teams, output_dim=30, input_length=2, init='uniform', trainable=True))
	model.add(Flatten())
	model.add(Dense(15))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model

def create_submission(model, team_to_int):
	sub = pd.read_csv('../input/sample_submission.csv')
	sub['team1'] = sub['id'].apply(lambda x: team_to_int[int(x.split('_')[1])])
	sub['team2'] = sub['id'].apply(lambda x: team_to_int[int(x.split('_')[2])])
	sub['pred'] = model.predict(sub[['team1','team2']].values, batch_size=1024, verbose=2)
	sub = sub[['id','pred']]
	sub.to_csv('submission.csv', index=False)

def main():
	train, team_to_int = process_data()
	num_teams = len(team_to_int.keys())
	model = create_model(num_teams)
	model.compile(Adam(0.001), loss='binary_crossentropy')
	model.fit(train[:,:2], train[:,2], batch_size=1024, nb_epoch=200, validation_split=0.1, verbose=2)
	create_submission(model, team_to_int)


if __name__ == '__main__': main()