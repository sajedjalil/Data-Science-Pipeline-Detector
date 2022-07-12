import pip
import pandas as pd

output_list = []
for package in sorted(pip.get_installed_distributions(), key=lambda x: x.project_name):
    print("{} ({})".format(package.project_name, package.version))
    output_list.append((package.project_name, package.version))
    
    
df = pd.DataFrame(output_list)
df.columns = ['pip_project_name','version']
print(df.head(3))
df.to_csv("submission.csv", index=False)

