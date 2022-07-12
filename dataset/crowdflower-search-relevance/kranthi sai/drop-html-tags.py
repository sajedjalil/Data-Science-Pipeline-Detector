
import pandas as pd
from bs4 import BeautifulSoup
import re

# Use Pandas to read in the training and test data
#train = pd.read_csv("../input/train.csv").fillna("")
#test  = pd.read_csv("../input/test.csv").fillna("")

# Print a sample of the training data
#print(train.head())

# Now it's yours to take from here!
text="<p><strong>This text is strong</strong>.</p>"

print(BeautifulSoup(text).get_text(separator=" "))
l="16  gb 32 gb 8 gb"
for vol in [16, 32, 64, 128, 500]:
    l = re.sub("%d gb"%vol, "%dgb"%vol, l)
    l = re.sub("%d g"%vol, "%dgb"%vol, l)
    l = re.sub("%dg "%vol, "%dgb "%vol, l)
print(l)