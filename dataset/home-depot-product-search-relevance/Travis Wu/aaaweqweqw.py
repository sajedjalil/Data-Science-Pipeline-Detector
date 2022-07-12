import pip

for dist in pip.get_installed_distributions():
    print(dist)