import pkg_resources
import platform

print('[Python version]')
print(platform.python_version())

print()

print('[Packages]')
for dist in pkg_resources.working_set:
    print(dist.project_name, dist.version)
