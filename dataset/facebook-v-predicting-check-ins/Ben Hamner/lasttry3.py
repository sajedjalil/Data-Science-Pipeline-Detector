import numpy as np

size = 10.0;

x_step = 0.2
y_step = 0.2

x_ranges = list(zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step)))
y_ranges = list(zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step)))

for x,y in x_ranges:
    print("x")
    print(x,y)
    for a,b in y_ranges:
        print("y")
        print(a,b)
