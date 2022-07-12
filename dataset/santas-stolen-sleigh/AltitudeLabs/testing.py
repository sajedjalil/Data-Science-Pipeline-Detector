import datetime

j=1
for i in range(1, 500000):
    j = j*i % 50 +1
    print(j)
print(datetime.datetime.now())