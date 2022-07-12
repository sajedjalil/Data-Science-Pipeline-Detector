from matplotlib import pyplot as plt

def get_readings_per_hour(filename):
    readings_per_hour = {}
    with open(filename) as f:
        f.readline()
        prev_hour_id = None
        cnt = 0
        for line in f:
            hour_id = int(line.split(',')[0])
            if hour_id != prev_hour_id:
                if cnt in readings_per_hour:
                    readings_per_hour[cnt] += 1
                else:
                    readings_per_hour[cnt] = 1
                cnt = 0
            cnt += 1
            prev_hour_id = hour_id
        readings_per_hour.pop(0)
    total_readings = sum(readings_per_hour.values())
    for key in readings_per_hour:
        readings_per_hour[key] /= float(total_readings)
    return readings_per_hour


histogram_test = get_readings_per_hour('../input/test.csv')
histogram_train = get_readings_per_hour('../input/train.csv')

plt.figure()
plt.title('Distribution of number of readings, test dataset')
plt.bar(histogram_test, histogram_test.values())
#ax[0].set_xticks(histogram_test.keys())
plt.xlabel('readings per hour')
plt.ylabel('percentage of observations')
plt.savefig('number_of_readings_test.png')

plt.figure()
plt.bar(histogram_train, histogram_train.values())
#ax[1].set_xticks(histogram_train.keys())
plt.title('train_dataset')
plt.xlabel('readings per hour')
plt.ylabel('percentage of observations')
plt.savefig('number_of_readings_train.png')