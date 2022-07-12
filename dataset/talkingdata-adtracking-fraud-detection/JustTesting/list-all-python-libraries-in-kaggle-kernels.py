import sys

print("Python version: " + sys.version)

print("\nConda version:")
!conda --version

print("\nPip version:")
!pip --version

print("\nConda environments:")
!conda info --envs

print("\nInstalled conda libraries in root environment:")
!conda list -n root

