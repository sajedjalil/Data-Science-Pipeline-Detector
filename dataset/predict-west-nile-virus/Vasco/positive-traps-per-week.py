import itertools as it

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

SPECIES = ['CULEX PIPIENS', 
          'CULEX PIPIENS/RESTUANS',
          'CULEX RESTUANS']

traps = pd.read_csv('../input/train.csv', parse_dates=['Date'])[['Species', 'WnvPresent', 'Date', 'Trap']]
traps = traps.groupby(['Date', 'Trap', 'Species']).max().reset_index()

traps['Year'] = traps['Date'].map(lambda x: x.year)
traps['Week'] = traps['Date'].map(lambda x: x.week)

traps = traps[['Year','Week','WnvPresent','Species']]
checks = traps[['Week', 'Year','WnvPresent']].groupby(['Week','Year']).count().reset_index()

weekly_postives = traps.groupby(['Year','Week', 'Species']).sum().reset_index()
weekly_postives_species = weekly_postives.set_index(['Year','Week', 'Species']).unstack()
weekly_postives_species.columns = weekly_postives_species.columns.get_level_values(1)
weekly_postives_species['total_positives'] = weekly_postives_species.sum(axis=1)
weekly_postives_species = weekly_postives_species.reset_index().fillna(0)
    
weekly_checks = checks.groupby(['Year','Week']).sum()
weekly_checks.columns = ['checks']
weekly_checks = weekly_checks.reset_index()
weekly_checks['positive'] = weekly_postives_species['total_positives']
weekly_checks['trap_infection_rate'] = weekly_checks['positive'] / weekly_checks['checks'] * 100
weekly_checks_years = weekly_checks.pivot(index='Week', columns='Year', values='trap_infection_rate')

ax = weekly_checks_years.interpolate().plot(title='Trap infection rate', figsize=(10,6))
ax.set_ylabel('Perctenage traps infected');
plt.savefig('postive_trap_rate.png')

fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, sharex=True)
fig.tight_layout()
axit = (ax for ax in it.chain(*axes))

for m, group in weekly_postives_species.groupby('Year'):
    ax = next(axit); ax.xaxis.grid(); ax.yaxis.grid()
    for species in SPECIES:
        ax.plot(group['Week'], group[species], label=species)
    ax.legend(loc='upper left');
    ax.set_title(m)
plt.savefig('postive_traps.png')