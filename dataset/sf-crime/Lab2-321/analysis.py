import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(34,79,100)
y = 57*np.sin(np.pi/2*(x-34)/45)**2

plt.plot(x,y)
plt.show()