# Credit to https://www.kaggle.com/users/114978/triskelion for originally
# creating this script on another competition
# (https://www.kaggle.com/users/114978/triskelion/bike-sharing-demand/python-test)

import sys
import pip
import warnings

warnings.filterwarnings("ignore") 

print(sys.version_info)
print()

for available_distro in sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]):
    print(available_distro)

help('modules')