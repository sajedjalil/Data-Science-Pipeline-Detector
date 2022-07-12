# fast submission - VALID FOR PUBLIC/PRIVATE LB
# pay attention also to
# https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/177662
# https://www.kaggle.com/corochann/save-your-time-submit-without-kernel-inference

# MOTIVATION: formally speaking, we are participating in kernel completition, 
# but reality is telling us different things - it's casual predict locally then submit csv comp

# OVERVIEW: this code will make:
# 1. create private dataset with your submission.csv file
# 2. create private kernel
# 3. attach dataset from step 1 to kernel from step 2
# 4. kernel will transfer submission.csv from step 1 as submission for completition

# PREPARATION
# before execution of this script you have to 
# 1. install kaggle python package
# 2. autorize in kaggle cli
# https://github.com/Kaggle/kaggle-api#api-credentials
# 3 put this code into "make_submit.py" in "make_submit" forlder in any place, which you wish

# HOWTO use
# python make_submit/make_submit.py --path relative_or_absolute_path/sub.csv
# you have to be sure that sub.csv has proper format for submission
import os
import argparse
import json
from pathlib import Path
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True,
    help="absolute or relative path to submission")
parser.add_argument(
    "--comment", type=str, required=False, default="default_comment",
    help="will be visible in kaggle GUI")
args = parser.parse_args()
submission_path = Path(args.path)


### cleanup folder before dataset creation
files_in_make_submit_folder = Path("make_submit").glob("*")
for filename in files_in_make_submit_folder:
    if filename.name != "make_submit.py":
        os.remove(str(filename))

### prepare dataset 
os.system("kaggle datasets init -p make_submit")
path = Path("make_submit/dataset-metadata.json")
text = path.read_text()
text = text.replace("INSERT_TITLE_HERE", "submission-dataset")
text = text.replace("INSERT_SLUG_HERE", "user-slug2")
path.write_text(text)
os.system("kaggle datasets create -p make_submit")
os.system(f"cp {submission_path} make_submit/submission.csv")
os.system(f"kaggle datasets version -p make_submit -m \"{args.comment}\"")


### create extremely complicated code for our kernel
os.system("touch make_submit/kernel_code.py")
path = Path("make_submit/kernel_code.py")
text = "import os; os.system(\"cp /kaggle/input/user-slug2/submission.csv ./submission.csv\")"
path.write_text(text)


### wait some time, because kaggle is processing our dataset
time.sleep(120)

### prepare kernel metadata 
os.system("kaggle kernels init -p make_submit")
with open("make_submit/kernel-metadata.json") as f:
    data = json.load(f)


username = data["id"].split("/")[0]

path = Path("make_submit/kernel-metadata.json")
text = path.read_text()
text = text.replace("INSERT_KERNEL_SLUG_HERE", "submission-kernel")
text = text.replace("INSERT_TITLE_HERE", "submission-kernel")
text = text.replace("INSERT_CODE_FILE_PATH_HERE", "kernel_code.py")
text = text.replace("Pick one of: {python,r,rmarkdown}", "python")
text = text.replace("Pick one of: {script,notebook}", "script")
text = text.replace(
    "\"dataset_sources\": []", 
    f"\"dataset_sources\": [\"{username}/user-slug2\"]")
text = text.replace(
    "\"competition_sources\": []", 
    f"\"competition_sources\": [\"lyft-motion-prediction-autonomous-vehicles\"]")
path.write_text(text)
os.system("kaggle kernels push -p make_submit")

### after all, you just need to go to the kernel page 
# "https://www.kaggle.com/your_username/submission-kernel" and press "submit" button
