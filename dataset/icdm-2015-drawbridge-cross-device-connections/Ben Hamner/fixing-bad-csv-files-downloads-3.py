
import csv
from io import StringIO
import subprocess

def fix_bad_csv(input_file, output_file, bad_col):
    with open(input_file) as f_in:
        with subprocess.Popen("gzip -c > %s" % output_file, shell=True, stdin=subprocess.PIPE) as p:
            for line in f_in:
                row = csv.reader(StringIO(line.strip().replace("{", "\"").replace("}", "\""))).__next__()
                for el in csv.reader(StringIO(row[bad_col].replace("(", "\"").replace(")", "\""))).__next__():
                    p.stdin.write(bytes(",".join(row[:bad_col] + el.split(",")) + "\n", "UTF-8"))
            p.communicate()

# fix_bad_csv("../input/property_category.csv", "property_category_corrected.csv.gz", 1) # 2nd column
# fix_bad_csv("../input/id_all_ip.csv", "id_all_ip.csv.gz", 2) # 3rd column
fix_bad_csv("../input/id_all_property.csv", "id_all_property.csv.gz", 2) # 3rd column
