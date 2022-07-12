import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f = open('../input/driver_imgs_list.csv')
lines = f.readlines()[1:] # header
drivers = {}
for line in lines:
	driver, c, img = line[:-1].split(',')
	if driver not in drivers:
		drivers[driver] = [0 for x in range(10)]
	classindex = int(c[1])
	drivers[driver][classindex] += 1
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow([drivers[i] for i in drivers])
fig.colorbar(cax)
ax.set_yticks(range(len(drivers.keys())))
ax.set_xticks(range(10))
ax.set_yticklabels(drivers.keys())
ax.set_xticklabels(range(10))
plt.savefig('drivers.png')
# Here is why random training/validation splitting doesn't work. Each driver has lots of images for every class which are very similar!