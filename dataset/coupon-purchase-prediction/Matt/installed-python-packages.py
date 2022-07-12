import pip
dists = sorted(["{}, {}".format(i.key, i.version) for i in pip.get_installed_distributions()])
print("\n".join(dists))
