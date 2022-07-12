import pkg_resources
for dist in pkg_resources.working_set:
    print(dist.project_name, dist.version)