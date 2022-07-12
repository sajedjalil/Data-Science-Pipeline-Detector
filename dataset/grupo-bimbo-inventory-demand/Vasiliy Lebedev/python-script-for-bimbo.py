import pandas as pd
train_data = pd.read_csv('../input/train.csv', nrows=500000)

pd.set_option('display.width', 256)

print(train_data.groupby(["Cliente_ID","Producto_ID"])["Venta_uni_hoy"].describe())