import os
import shutil

import pandas as pd
import pydicom
from tqdm import tqdm

INPUT_FOLDER = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
DEPS_FOLDER = "/kaggle/input/vinbigdatachestxraydeps"
WORK_FOLDER = "/kaggle/working"

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")


def copy_deps():
    if not DEVMODE:
        for file_name in os.listdir(DEPS_FOLDER):
            shutil.copy(os.path.join(DEPS_FOLDER, file_name), os.path.join(WORK_FOLDER, file_name))


def get_dicom_tags(file_path, tags_filter=None):
    dcm = pydicom.read_file(file_path)
    tags = {}
    for key in dcm.keys():
        if tags_filter is not None and dcm[key].name not in tags_filter:
            continue
        if dcm[key].name == "Pixel Data":
            continue
        tags[dcm[key].name] = dcm[key].value
    return tags


def main():
    copy_deps()

    dependencies_dir = "vinbigdata-chest-xray-abnormalities-detection" if DEVMODE else DEPS_FOLDER
    dicom_metadata_desc = pd.read_csv(os.path.join(dependencies_dir, "dicomMetadata.csv"))
    tags_names = list(dicom_metadata_desc["Attribute Name"].values)

    for set_name in ["train", "test"]:
        cur_set_folder = os.path.join(INPUT_FOLDER, set_name)

        metadata = pd.DataFrame(columns=["image_id"] + tags_names)

        for file_name in tqdm(os.listdir(cur_set_folder)):
            image_id = os.path.splitext(file_name)[0]
            cur_file_path = os.path.join(cur_set_folder, file_name)

            tags = get_dicom_tags(cur_file_path, tags_filter=tags_names)

            metadata = metadata.append({**tags, "image_id": image_id}, ignore_index=True)

        print(metadata.head())
        metadata.to_csv(f"{set_name}_metadata.csv", index=False)


if __name__ == "__main__":
    main()
