import subprocess
subprocess.check_call(["python", '-m', 'pip', 'install',"--upgrade", 'numpy==1.17.3'])
import numpy as np

print('> NumPy version: {}'.format(np.__version__))