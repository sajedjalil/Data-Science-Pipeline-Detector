from matplotlib import pyplot as plt
from pylab import savefig
import seaborn as sns

print('Started')
i = 0
r_sum = [0] * 24
r_cnt = [0] * 24
names = []
with open("../input/train_ver2.csv") as infile:
    for line in infile:
        line = line[:-1] #newline
        i += 1
        if i == 1: #header
            names = line.split(",")[24:]
            continue 
    
        tmp = line.split("\"")
        arr = tmp[0][:-1].split(",") + [tmp[1]] + tmp[2][1:].split(',')

        products = arr[24:]

        renta = arr[22].strip()
        if renta == '' or renta == 'NA':
            renta = 0
        else:
            renta = float(renta)

        for p in range(24):
            if products[p].strip() == '1':
                r_sum[p] += renta
                r_cnt[p] += 1

        if i % 1000000 == 0:
            print('%s lines processed' % str(i))

print("Results:")
res = [0] * 24
for p in range(24):
    names[p] = names[p].replace("ind_", "").replace("_ult1", "")
    if r_cnt[p] > 0:
        res[p] = round(r_sum[p] / r_cnt[p])
        print('%s - average income: %s' % (names[p], str(res[p])))
    else:
        print('%s - no data' % names[p])

plt.figure(figsize=(8,8))
sns.barplot(names, res, alpha=0.8, color=sns.color_palette()[0])
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Average income', fontsize=12)
plt.xticks(rotation='vertical')
savefig('average_income.png')
plt.gcf().clear()
        
        
