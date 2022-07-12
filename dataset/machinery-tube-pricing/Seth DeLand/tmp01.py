import os
os.system("ls ../input")
os.system("echo \n\n")
#os.system("head ../input/*")

import pandas as pd

bom = pd.read_csv("../input/bill_of_materials.csv");
tube = pd.read_csv("../input/tube.csv");

bom = bom.join(tube,on='tube_assembly_id',rsuffix='_tube');
print(bom.head(5).values)