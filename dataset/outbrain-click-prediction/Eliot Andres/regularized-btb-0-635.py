
import pandas as pd
reg = 10


for line in open("../input/events.csv"):
 if "\\N" in line:
   print(line)