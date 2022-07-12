from pip._internal.utils.misc import get_installed_distributions

for package in sorted(get_installed_distributions(), key=lambda x: x.project_name):
    print("{} ({})".format(package.project_name, package.version))
