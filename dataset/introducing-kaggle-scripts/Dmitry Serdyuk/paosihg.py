
with open('test.csv', 'w') as f:
    f.write("id, feat1, feat2\n")
    f.write("1, 12, 2\n")
    f.write("2, 1, 2\n")
    f.write("3, 16, 2\n")
    f.write("4, 2, 2\n")
    f.write("5, 112, 2\n")
    f.write("6, 12, 2\n")
    
from matplotlib import pyplot as plt

plt.plot([1, 2, 3], [5, 1, 5], 'r')
plt.savefig('hello.png')
plt.show()