from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [12, 10]
plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(range(4), [4, 6, 40, 50], 0.6)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(4) + 0.3)
ax.set_ylim([0, 60])
ax.set_xticklabels(['READ\nFORUM',
                    'GIVE UP\nAND\nBOOKMARK POSTS',
                    'THINK ABOUT HOW\nIS IT EVEN POSSIBLE\nTO REACH 0.6 MAP@5??',
                    'REFRESH THE\nLEADERBOARD\nKEEP CALM\nAND CARRY ON'])
plt.ylabel('TIME')
plt.yticks([])
plt.title("MY LAST WEEK IN THE COMPETITION")
plt.show()
plt.savefig('last_week_xkcd.png')
