import pip

for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):
    print("{} ({})".format(package.project_name, package.version))