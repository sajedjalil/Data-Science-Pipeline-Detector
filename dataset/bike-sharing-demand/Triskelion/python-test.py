import sys
import pip
import warnings
import seaborn # :)

warnings.filterwarnings("ignore") # only once, I promise.

print(sys.version_info)
print()

for available_distro in sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]):
    print(available_distro)

help('modules')