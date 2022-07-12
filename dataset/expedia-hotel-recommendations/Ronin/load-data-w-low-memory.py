

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv", dtype={'id':np.int64,
                                                    'date_time':object,
                                                    'site_name':np.uint8,
                                                    'posa_continent':np.uint8,
                                                    'user_location_country':np.uint8,
                                                    'user_location_region':np.int16,
                                                    'user_location_city':np.int32,
                                                    'orig_destination_distance':np.float32,
                                                    'user_id':np.int32,
                                                    'is_mobile':bool,
                                                    'is_package':bool,
                                                    'channel':np.int8,
                                                    'srch_ci':np.object,
                                                    'srch_co':np.object,
                                                    'srch_adults_cnt':np.uint8,
                                                    'srch_children_cnt':np.uint8,
                                                    'srch_rm_cnt':np.uint8,
                                                    'srch_destination_id':np.uint16,
                                                    'srch_destination_type_id':np.int8,
                                                    'hotel_continent':np.int16,
                                                    'hotel_country':np.uint8,
                                                    'hotel_market':np.int16,
                                                    'is_booking':bool,
                                                    'cnt':np.int16,
                                                    'hotel_cluster':np.int16})
