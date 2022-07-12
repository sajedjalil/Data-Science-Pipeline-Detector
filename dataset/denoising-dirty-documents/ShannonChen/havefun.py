import os
from PIL import Image

os.system("ls ../input")

os.system('''echo "\n\nLet's look inside the directories:"''')
os.system("ls ../input/*")

# Copy a couple images to working so they are displayed as output:

os.system("cp ../input/train/101.png train_101.png")
#os.system("cp ../input/train_cleaned/101.png train_cleaned_101.png")

jpg = Image.open('train_101.png')

print(jpg.getdata())