import numpy as np
import pandas as pd

# Full table:   6.1Gb
# This version: 1.1Gb (-82%)
types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,
         'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16,
         'Demanda_uni_equil':np.uint32}

a = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)
print(a.info(memory_usage=True))