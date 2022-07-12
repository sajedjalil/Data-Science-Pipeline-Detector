from __future__ import print_function
import pip

with open('installed.txt', 'w') as out:
    for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):
      print("{} ({})".format(package.project_name, package.version))
      print("{} ({})".format(package.project_name, package.version), file=out)
