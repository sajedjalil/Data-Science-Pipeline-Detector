# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 

tourney = pd.read_csv('../input/TourneyDetailedResults.csv')
regularSeason = pd.read_csv('../input/RegularSeasonDetailedResults.csv')

frames = [tourney, regularSeason]
combined = pd.concat(frames)

combined = combined[combined.Season >= 2012]
combined = combined[combined.Season <= 2015]

combined = combined.drop(['Daynum', 'Wloc', 'Numot'], 1)
combined['score'] = combined.Wscore - combined.Lscore
combined['fgm'] = combined.Wfgm - combined.Lfgm
combined['fga'] = combined.Wfga - combined.Lfga
combined['fgm3'] = combined.Wfgm3 - combined.Lfgm3
combined['fga3'] = combined.Wfga3 - combined.Lfga3
combined['ftm'] = combined.Wftm - combined.Lftm
combined['fta'] = combined.Wfta - combined.Lfta
combined['or'] = combined.Wor - combined.Lor
combined['dr'] = combined.Wdr - combined.Ldr
combined['ast'] = combined.Wast - combined.Last
combined['to'] = combined.Wto - combined.Lto
combined['stl'] = combined.Wstl - combined.Lstl
combined['blk'] = combined.Wblk - combined.Lblk
combined['pf'] = combined.Wpf - combined.Lpf
combined = combined.drop(['Wscore', 'Lscore', 'Wfgm', 'Lfgm', 'Wfga', 'Lfga', 'Wfgm3', 'Lfgm3', 'Wfga3', 'Lfga3', 'Wftm', 'Lftm', 'Wfta', 'Lfta', 'Wor', 'Lor', 'Wdr', 'Ldr', 'Wast', 'Last', 'Wto', 'Lto', 'Wstl', 'Lstl', 'Wblk', 'Lblk', 'Wpf', 'Lpf'], 1)

combined.to_csv('combinedData.csv', index=False)

# Any results you write to the current directory are saved as output.