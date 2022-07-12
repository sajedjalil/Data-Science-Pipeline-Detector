from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 9]
plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(range(3), [44, 57, 60], 0.6)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(3) + 0.3)
ax.set_ylim([0, 60])
ax.set_xticklabels(['Heard about Canada Yan?',
                    'Qiang(3) Po Zheng',
                    'Ok let us try, \n since you are always right!'
                    ])
plt.ylabel('Never bow down to the Evil')
plt.yticks([])
plt.title("Talking about Histogram I only fu you.")
plt.show()
plt.savefig('lalalalala.png')



