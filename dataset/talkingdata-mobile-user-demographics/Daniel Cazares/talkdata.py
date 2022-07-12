# Input data files are available in the "../input/" directory.
import numpy as np
import pandas as pd

app_events = pd.read_csv("../input/app_events.csv")
app_labels = pd.read_csv("../input/app_labels.csv")
events = pd.read_csv("../input/events.csv")
label_categories = pd.read_csv("../input/label_categories.csv")
phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv")

gender_age_train = pd.read_csv("../input/gender_age_train.csv")

# phone_brand_device_model.info()
deduped_phone_brand_device_model = phone_brand_device_model.drop_duplicates()

gender_age_train.info()