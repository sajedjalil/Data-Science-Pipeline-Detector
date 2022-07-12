import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,2,0.01)*np.pi
y = np.sin(x)
plt.plot(x,y)
plt.savefig('sin.png', bbox_inches='tight')
with open('hello.html','w') as f:
    f.write('<img src="sin.png">')
    
