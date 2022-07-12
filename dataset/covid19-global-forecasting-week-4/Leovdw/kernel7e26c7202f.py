# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
# import country
# import geounit
# import datapoint
import numpy as np
import datetime
from keras import models
from keras import layers
from keras.models import load_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

class Country:
    def __init__(self, region_id, country_name, zone, population, area, density, cost_area_ratio, migration,
                 inf_mortality,
                 gdp, literacy, phones, arable, crops, other, climate, birthrate, death_rate, agriculture,
                 industry, service):
        self.region_id = int(region_id)
        self.country_name = country_name.strip()
        self.zone = float(zone)
        self.population = Country.comma_to_dot(self,population)
        self.area = Country.comma_to_dot(self,area)
        self.density = Country.comma_to_dot(self,density)
        self.cost_area_ratio = Country.comma_to_dot(self,cost_area_ratio)
        self.migration = Country.comma_to_dot(self,migration)
        self.inf_mortality = Country.comma_to_dot(self,inf_mortality)
        self.gdp = Country.comma_to_dot(self,gdp)
        self.literacy = Country.comma_to_dot(self,literacy)
        self.phones = Country.comma_to_dot(self,phones)
        self.arable = Country.comma_to_dot(self,arable)
        self.crops = Country.comma_to_dot(self,crops)
        self.other = Country.comma_to_dot(self,other)
        self.climate = Country.comma_to_dot(self,climate)
        self.birthrate = Country.comma_to_dot(self,birthrate)
        self.death_rate = Country.comma_to_dot(self,death_rate)
        self.agriculture = Country.comma_to_dot(self,agriculture)
        self.industry = Country.comma_to_dot(self,industry)
        self.service = Country.comma_to_dot(self,service)
        self.region_id = 0
        self.region_name = ""
        self.outbreak_day = 0
        self.past_threshold_day = -1
        self.index_first_test = 0
        Country.improve_data(self)

    def improve_data(self):
        if self.country_name == "United States":
            self.country_name = 'US'

    def comma_to_dot(self, stringvalue):
        if len(stringvalue) == 0 or stringvalue.isspace():
            return float(0)
        s = stringvalue.replace(",", ".")
        return float(s)

    def get_country_name(self):
        return self.country_name

    def get_zone(self):
        return self.zone

    def get_region(self):
        return self.region_name

    def get_region_id(self):
        return self.region_id

    def get_population(self):
        return self.population

    def get_area(self):
        return self.area

    def get_density(self):
        return self.density

    def get_cost_area_ratio(self):
        return self.cost_area_ratio

    def get_migration(self):
        return self.migration

    def get_inf_mortality(self):
        return self.inf_mortality

    def get_gdp(self):
        return self.gdp

    def get_literacy(self):
        return self.literacy

    def get_phones(self):
        return self.phones

    def get_arable(self):
        return self.arable

    def get_crops(self):
        return self.crops

    def get_other(self):
        return self.other

    def get_climate(self):
        return self.climate

    def get_birthrate(self):
        return self.birthrate

    def get_death_rate(self):
        return self.death_rate

    def get_agriculture(self):
        return self.agriculture

    def get_industry(self):
        return self.industry

    def get_service(self):
        return self.service

class Data_point:
    def __init__(self, id, region_number, julian_day_number, past_outbreak_day_number, past_threshold_days, cases,
                 deaths, hist_cases, hist_fatalities):
        self.id = id
        self.region_number = region_number
        self.julian_day_number = julian_day_number
        self.days_since_outbreak = past_outbreak_day_number
        self.cases = cases
        self.deaths = deaths
        self.num_cases = len(hist_cases)
        self.hist_cases = np.zeros(self.num_cases, dtype=np.float32)
        for i in range(self.num_cases):
            self.hist_cases[i] = hist_cases[i]
        self.num_fatalities = len(hist_fatalities)
        self.hist_fatalities = np.zeros(self.num_fatalities, dtype=np.float32)
        for i in range(self.num_fatalities):
            self.hist_fatalities[i] = hist_fatalities[i]
        self.days_since_threshold = past_threshold_days
        self.zone = np.float32(0)
        Data_point.improve_data(self)

    def get_id(self):
        return self.id

    def get_cases(self):
        return self.cases

    def get_deaths(self):
        return self.deaths

    def get_region_number(self):
        return self.region_number

    def improve_data(self):
        pass

    def get_days_since_outbreak(self):
        return self.days_since_outbreak

    def get_julian_day_number(self):
        return self.julian_day_number

    def set_days_since_threshold(self, n):
        self.days_since_threshold = n
        return

    def get_days_since_threshold(self):
        return self.days_since_threshold

    def set_zone(self, zn):
        self.zone = np.float32(zn)
        return

    def get_zone(self):
        return self.zone

    def get_history_item(self, n):
        return self.history[n]

    def get_datapoint_np(self, num_features):
        p = np.zeros(num_features, dtype=np.float32)
        for i in range(num_features):
            if i == 0:
                p[0] = self.get_days_since_threshold()
            if i == 1:
                p[1] = self.get_days_since_outbreak()
            if 2 <= i < self.num_cases + 2:
                p[i] = self.hist_cases[i - 2]
            if self.num_cases + 2 <= i < self.num_fatalities + self.num_cases + 2:
                p[i] = self.hist_fatalities[i - self.num_cases - 2]
        return p

    def shiftleft_cases(self, next_cases, next_fatalities):
        for i in range(self.num_cases - 1):
            self.hist_cases[i] = self.hist_cases[i + 1]
        self.hist_cases[self.num_cases - 1] = np.float32(next_cases)
        for i in range(self.num_fatalities - 1):
            self.hist_fatalities[i] = self.hist_fatalities[i + 1]
        self.hist_fatalities[self.num_fatalities - 1] = np.float32(next_fatalities)
        return

class Geo_unit:
    def __init__(self, region_id, region_name, country_name):
        self.region_id = int(region_id)
        self.country_name = country_name.strip()
        self.region_name = region_name.strip()
        self.zone = 0
        self.population = 0
        self.area = 0
        self.density = 0
        self.cost_area_ratio = 0
        self.migration = 0
        self.inf_mortality = 0
        self.gdp = 0
        self.literacy = 0
        self.phones = 0
        self.arable = 0
        self.crops = 0
        self.other = 0
        self.climate = 0
        self.birthrate = 0
        self.death_rate = 0
        self.agriculture = 0
        self.industry = 0
        self.service = 0
        self.outbreak_day = 0
        self.past_threshold_day = -1
        self.index_first_test = 0
        self.special_interest = False
        # self.special_interest = True
        self.problem_area = 0
        self.last_spreading = 0
        self.lower_limit_cases = 0              #20200414
        self.lower_limit_deaths = 0             #20200414
        Geo_unit.improve_data(self)


    def complement(self, zone, population, area, density, cost_area_ratio, migration, inf_mortality,
                   gdp, literacy, phones, arable, crops, other, climate, birthrate, death_rate, agriculture,
                   industry, service):
        self.zone = np.float32(zone)
        self.population = np.float32(population)
        self.area = np.float32(area)
        self.density = np.float32(float(density))
        self.cost_area_ratio = np.float32(cost_area_ratio)
        self.migration = np.float32(migration)
        self.inf_mortality = np.float32(inf_mortality)
        self.gdp = np.float32(gdp)
        self.literacy = np.float32(literacy)
        self.phones = np.float32(phones)
        self.arable = np.float32(arable)
        self.crops = np.float32(crops)
        self.other = np.float32(other)
        self.climate = np.float32(climate)
        self.birthrate = np.float32(birthrate)
        self.death_rate = np.float32(death_rate)
        self.agriculture = np.float32(agriculture)
        self.industry = np.float32(industry)
        self.service = np.float32(service)
        self.index_first_test = 0
        self.special_interest = False
        Geo_unit.improve_data(self)

    def improve_data(self):
        if self.country_name == "United States":
            self.country_name = 'US'
        interesting = ("California", "Colombia",  "Louisiana", "Netherlands", "New York","Spain", "Washington")
        if self.region_name in interesting:
            Geo_unit.set_special_interest(self)
            text = "{} gets my special attention in zone {}."
            print(text.format(self.region_name, self.zone))

    def get_country_name(self):
        return self.country_name

    def set_outbreak_day(self, fecha):
        self.outbreak_day = int(fecha.timetuple().tm_yday)
        return

    def get_outbreak_day(self):
        return self.outbreak_day

    def set_past_threshold_day(self, fecha):
        self.past_threshold_day = int(fecha.timetuple().tm_yday)
        # text = "Outbreak day in {} was {} and exceeded the threshold on {}."
        # print(text.format(self.region_name, self.outbreak_day, self.past_threshold_day))
        return

    def get_past_threshold_day(self):
        return self.past_threshold_day

    def get_zone(self):
        return self.zone

    def get_geounit(self, num_features):
        p = np.zeros(num_features)
        for i in range(num_features):
            if i == 0: p[i] = self.get_zone()
            if i == 1: p[i] = self.get_density()
            if i == 2: p[i] = self.get_population()
            if i == 3: p[i] = self.get_climate()
            if i == 4: p[i] = self.get_gdp()
            if i == 5: p[i] = self.get_inf_mortality()
            if i == 6: p[i] = self.get_death_rate()
            if i == 7: p[i] = self.get_birthrate()
        return p

    def get_region(self):
        return self.region_name

    def get_region_id(self):
        return self.region_id

    def get_density(self):
        return self.density

    def get_population(self):
        return self.population

    def get_climate(self):
        return self.climate

    def get_gdp(self):
        return self.gdp

    def get_inf_mortality(self):
        return self.inf_mortality

    def get_death_rate(self):
        return self.death_rate

    def get_birthrate(self):
        return self.birthrate

    def set_special_interest(self):
        self.special_interest = True
        return

    def get_special_interest(self):
        return self.special_interest

    def set_problem_area(self, problem_indicator):
        self.problem_area = problem_indicator
        return

    def get_problem_area(self):
        return self.problem_area

    def set_last_spreading(self, last_spreading):
        self.last_spreading = last_spreading
        return

    def get_last_spreading(self):
        return self.last_spreading

    def set_lower_values(self, cases, deaths):                      # 20200414
        self.lower_limit_cases = cases                              # 20200414
        self.lower_limit_deaths = deaths                            # 20200414
        return                                                      # 20200414

    def set_lower_value_cases(self, cases):                      # 20200414
        self.lower_limit_cases = cases                              # 20200414
        return                                                      # 20200414

    def set_lower_value_deaths(self, deaths):                      # 20200414
        self.lower_limit_deaths = deaths                            # 20200414
        return                                                      # 20200414

    def get_lower_value_cases(self):                                # 20200414
        return self.lower_limit_cases                               # 20200414

    def get_lower_value_deaths(self):                               # 20200414
        return self.lower_limit_deaths                              # 20200414


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(hidden_layers, activation='relu', input_shape=(num_features,)))
    model.add(layers.Dense(hidden_layers, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def production_train(optimum_epochs, input_data, input_labels_cases, test_data, test_labels, fname):
    create_model()
    model = create_model()
    model.fit(input_data, input_labels_cases, epochs=optimum_epochs, batch_size=16, verbose=0)
    model.save(fname + ".h5")
    model.save_weights(fname + "weightsonly.h5")
    results = model.evaluate(test_data, test_labels, verbose=2)
    print(results)
    return model


def test_train_model(k_fold, max_epochs, training_data, training_labels):
    assert k_fold >= 3, "The k_fold number should be at least 3 and no more than 5"
    assert k_fold <= 7, "The k_fold number should be at least 3 and no more than 7"
    fold_size = int(len(training_data) * (1 / k_fold))
    all_mae_histories = []
    for i in range(k_fold):
        model = create_model()
        validation_data = training_data[fold_size * i: fold_size * (i + 1)]
        validation_labels = training_labels[fold_size * i:fold_size * (i + 1)]
        partial_training_data = np.concatenate(
            [training_data[:fold_size * i], training_data[fold_size * (i + 1):]], axis=0)
        partial_training_labels = np.concatenate(
            [training_labels[:fold_size * i], training_labels[fold_size * (i + 1):]], axis=0)
        validate_from = fold_size * i
        validate_to = fold_size * (i + 1)
        data_from_1 = 0
        data_from_2 = fold_size * (i + 1)
        data_to_1 = fold_size * i
        data_to_2 = len(training_data)
        text = "Processing fold {}, data from {} to {} and {} to {}. Validation from {} to {}."
        print(text.format(i, data_from_1, data_to_1, data_from_2, data_to_2, validate_from, validate_to))
        history = model.fit(partial_training_data, partial_training_labels, epochs=max_epochs, batch_size=16,
                            validation_data=(validation_data, validation_labels), verbose=0)
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)
        del model
        del history
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(max_epochs)]
    smooth_mae_history = smooth_curve(average_mae_history[5:])
    # plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    # plt.xlabel("Epochs")
    # plt.ylabel("Validation MAE")
    # plt.show()
    optimae = 99999
    optimum = 0
    for i in range(len(average_mae_history)):
        if average_mae_history[i] < optimae:
            optimum = i
            optimae = average_mae_history[i]
    text = "Optimal number of epochs is {} with an MAE of {}"
    print(text.format(optimum, optimae))
    return optimum + 1


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def load_countries_of_the_world(fn):
    global zones, countries
    i = 0
    with open(fn, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if i > 0:
                region = row[1]
                if region in zones:
                    zone = zones.index(region)
                else:
                    zones.append(region)
                    zone = zones.index(region)
                g = Country(i - 1, row[0], zone, row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                                    row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16],
                                    row[17], row[18], row[19])
                countries.append(g)
            i += 1
    return i


def complement_data():
    global geounits
    for g in geounits:
        if g.get_country_name() in country_index:
            cptr = country_index.index(g.get_country_name())
            pais = countries[cptr]
            zone = pais.get_zone()
            population = pais.get_population()
            area = pais.get_area()
            density = pais.get_density()
            cost_area_ratio = pais.get_cost_area_ratio()
            migration = pais.get_migration()
            inf_mortality = pais.get_inf_mortality()
            gdp = pais.get_gdp()
            literacy = pais.get_literacy()
            phones = pais.get_phones()
            arable = pais.get_arable()
            crops = pais.get_crops()
            other = pais.get_other()
            climate = pais.get_climate()
            birthrate = pais.get_birthrate()
            death_rate = pais.get_death_rate()
            agriculture = pais.get_agriculture()
            industry = pais.get_industry()
            service = pais.get_service()
            g.complement(zone, population, area, density, cost_area_ratio, migration, inf_mortality,
                         gdp, literacy, phones, arable, crops, other, climate, birthrate, death_rate, agriculture,
                         industry, service)
        else:
            text = "{} is not in the countries table."
            print(text.format(g.get_country_name()))
            pass


def load_data(fn):
    global region_id, geounits, testpoints, history_cases, history_deaths
    region_id = -1
    i = 0
    previous_region = "x"
    previous_cases = 0
    max_info = 0
    with open(fn, newline='') as f:
        reader = csv.reader(f)
        g = "placeholder"
        history_cases = np.zeros(days_case_history, dtype=np.float32)
        history_deaths = np.zeros(days_deaths_history, dtype=np.float32)
        for row in reader:
            if i > 0:
                id = row[0]
                country = row[2]
                if row[1] == "":
                    region = country
                else:
                    region = row[1]
                ano = int(row[3][0:4])
                mes = int(row[3][5:7])
                dia = int(row[3][8:10])
                fecha = datetime.datetime(ano, mes, dia)
                cases = np.float32(row[4])
                deaths = np.float32(row[5])
                if previous_region != region:
                    region_id = region_id + 1
                    g = Geo_unit(region_id, region, country)
                    geounits.append(g)
                    previous_region = region
                    previous_cases = 0
                    previous_deaths = 0
                    history_cases = np.zeros(days_case_history, dtype=np.float32)
                    history_deaths = np.zeros(days_deaths_history, dtype=np.float32)
                else:
                    for i in range(days_case_history - 1):
                        history_cases[i] = history_cases[i + 1]
                    history_cases[days_case_history - 1] = previous_cases
                    for i in range(days_deaths_history - 1):
                        history_deaths[i] = history_deaths[i + 1]
                    history_deaths[days_deaths_history - 1] = previous_deaths
                if previous_cases == 0 and cases > 0:
                    g.set_outbreak_day(fecha)
                if previous_cases < threshold <= cases:
                    g.set_past_threshold_day(fecha)
                previous_cases = cases
                previous_deaths = deaths
                if cases >= 0:
                    delta = 0
                    if cases > 0:
                        delta = int(fecha.timetuple().tm_yday - g.get_outbreak_day())
                        if int(fecha.timetuple().tm_yday) <= first_test_day:            # 20200414
                            g.set_lower_values(cases, deaths)                           # 20200414
                    if delta > max_info:
                        max_info = delta
                    delta2 = -1
                    if g.get_past_threshold_day() != -1:
                        delta2 = int(fecha.timetuple().tm_yday - g.get_past_threshold_day())
                    d = Data_point(id, region_id, int(fecha.timetuple().tm_yday), delta, delta2,
                                                            cases, deaths, history_cases, history_deaths)
                    datapoints.append(d)
                    if int(fecha.timetuple().tm_yday) >= 93:
                        testpoints.append(d)
            i += 1
    f.close()
    return max_info


def prepare_data(days, ftd):
    global input_data, forecast_id, input_labels_cases, input_labels_fatalities
    row = 0
    test_row = 0
    for d in datapoints:
        rn = int(d.get_region_number())
        g = geounits[rn]
        g_data = g.get_geounit(8)
        cn = g.get_region()
        zn = int(g.get_zone())
        jn = int(d.get_julian_day_number())
        sn = int(g.get_outbreak_day())
        pt = int(g.get_past_threshold_day())
        # obd = jn - sn
        obd = d.get_days_since_outbreak()
        # if pt == -1:
        #     ptd = -1
        # else:
        #     ptd = jn - pt
        ptd = d.get_days_since_threshold()
        cs = d.get_cases()
        ft = d.get_deaths()
        if ptd >= 0:
            input_data[row, 0:data_point_features] = d.get_datapoint_np(data_point_features)
            input_data[row, data_point_features:] = g.get_geounit(geo_unit_features)
            input_labels_cases[row] = d.get_cases()
            input_labels_fatalities[row] = d.get_deaths()
            if g.get_special_interest():
                text = "\r\n{}: Cases: {} Deaths: {}. Data: "
                text = text.format(cn, input_labels_cases[row], input_labels_fatalities[row])
                for i in range(num_features):
                    text += " {}"
                    text = text.format(input_data[row, i])
                print(text)
            if jn >= ftd:
                test_data[test_row] = input_data[row]
                test_labels_cases[test_row] = input_labels_cases[row]
                test_labels_fatalities[test_row] = input_labels_fatalities[row]
                test_row += 1
            row += 1
    return row, test_row


#
#   Main line of the program starts here
#   1: define parameters.
#


print("Starting....")
threshold = 10  # I take the day a country/region exceeds this number of cases as the starting point
data_point_features = 2  # see the corresponding class to see which features will be included
days_case_history = 15     # number of days of cases to include in network
days_deaths_history = 15   # number of days of cases to include in network
geo_unit_features = 7  # see the corresponding class to see which features will be included
first_test_day = 93  # Julian day number (in 2020) of first row in test and submission data sets
days_to_predict = 43  # Number of days to include for each geounit in submission.csv
#
data_point_features = data_point_features + days_case_history + days_deaths_history
num_features = data_point_features + geo_unit_features      # number of columns to include in input_data
hidden_layers = 64           # 64
folding = 5
m_epochs = 350               # 250
use_saved_models = False
optimum_epochs = 0            # this will be calculated
includeCountry = False        # include the country/region name in the csv file for easy revision
includeCases = True           # must always be True, because the model works only uses cases to predict cases and fatalities
includeFatalities = True      # For testing, it may be False
#   2: define global variables
#
zones = []
countries = []  # Country level data
geounits = []
datapoints = []  # detail records by country/region and day
testpoints = []
forecast_id = 0
#
#   Write header of the finala output file
#
f = open("submission.csv", "w")
text = "ForecastId,ConfirmedCases,Fatalities\n"
f.write(text)
f.close()

#   2: load support datasets
#
n = load_countries_of_the_world('/kaggle/input/countries-of-the-world/countries of the world.csv')
text = "\r\n{} countries have been loaded."
print(text.format(n))
country_index = []
for c in countries:
    country_index.append(c.get_country_name())
#
outbreakdays = load_data('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
text = "\r\nThe maximum number of records for one country is {}."
print(text.format(outbreakdays))
complement_data()
# #
# #   2: Do some analysis on the data to check against results later on
# #
# analyse_input(outbreakdays)
#
#   3: Prepare a matrix for the neural networks
#
num_geo_units = len(geounits)
input_data = np.zeros((outbreakdays * num_geo_units, num_features), dtype=np.float32)
input_labels_cases = np.zeros(outbreakdays * num_geo_units, dtype=np.float32)
input_labels_fatalities = np.zeros(outbreakdays * num_geo_units, dtype=np.float32)
test_data = np.zeros((num_geo_units * 15, num_features), dtype=np.float32)
test_labels_cases = np.zeros(num_geo_units * 15, dtype=np.float32)
test_labels_fatalities = np.zeros(num_geo_units * 15, dtype=np.float32)
number_of_training_rows, number_of_test_rows = prepare_data(outbreakdays, first_test_day)
text = "Number of geographical units created is {}."
print(text.format(num_geo_units))
text = "Shape of input_data is {}"
print(text.format(input_data.shape))
training_size = int(number_of_training_rows * 1)
text = "Number of rows with useful training data is {}, of which we will use {}."
print(text.format(number_of_training_rows, training_size))
text = "Number of rows with useful testing data is {}. We will use them all"
print(text.format(number_of_test_rows))
input_data = np.resize(input_data, (number_of_training_rows, num_features))
input_labels_cases = np.resize(input_labels_cases, number_of_training_rows)
input_labels_fatalities = np.resize(input_labels_fatalities, number_of_training_rows)
test_data = np.resize(test_data, (number_of_test_rows, num_features))
test_labels_cases = np.resize(test_labels_cases, number_of_test_rows)
test_labels_fatalities = np.resize(test_labels_fatalities, number_of_test_rows)
text = "Shape of input_data is now {}"
print(text.format(input_data.shape))
text = "Shape of test_data is now {}"
print(text.format(test_data.shape))
# print(input_data)
# print(test_data)
mean = input_data[:training_size].mean(axis=0)
input_data -= mean
test_data -= mean
std = input_data[:training_size].std(axis=0)
input_data /= std
test_data /= std
if use_saved_models:
    print("Using saved models\r\n")
    production_model_cases = load_model("CoronaCases.h5")
    production_model_fatalities = load_model("CoronaFatalities.h5")
else:
    if includeCases:
        optimum_epochs = test_train_model(folding, m_epochs, input_data, input_labels_cases)
        production_model_cases = production_train(optimum_epochs, input_data, input_labels_cases, test_data,
                                                  test_labels_cases, "CoronaCases")
    else:
        pass
    if includeFatalities:
        optimum_epochs = test_train_model(folding, m_epochs, input_data, input_labels_fatalities)
        production_model_fatalities = production_train(optimum_epochs, input_data, input_labels_fatalities, test_data,
                                                       test_labels_fatalities, "CoronaFatalities")
    else:
        pass
of = open("submission.csv", "a")
row = 1
for d in datapoints:
    if d.get_julian_day_number() == first_test_day:
        for i in range(days_to_predict):
            input_data = np.zeros((1, num_features), dtype=np.float32)
            input_data[0, 0:data_point_features] = d.get_datapoint_np(data_point_features)
            g = geounits[d.get_region_number()]
            input_data[0, data_point_features:] = g.get_geounit(geo_unit_features)
            input_data -= mean
            input_data /= std
            predicted_cases = production_model_cases.predict(input_data, verbose=2)
            predicted_fatalities = production_model_fatalities.predict(input_data)
            next_cases = abs(predicted_cases[0, 0])                 # there should not be negative values, but there are
            next_fatalities = abs(predicted_fatalities[0, 0])       # there should not be negative values, but there are
            next_cases = max(next_cases, g.get_lower_value_cases())             # 20200414
            next_fatalities = max(next_fatalities, g.get_lower_value_deaths())  # 20200414
            d.shiftleft_cases(next_cases, next_fatalities)
            if includeCountry:
                text = "{},{},{},{}\n"
                text = text.format(row, int(next_cases), int(next_fatalities), g.get_region())
            else:
                text = "{},{},{}\n"
                text = text.format(row, int(next_cases), int(next_fatalities))
            of.write(text)
            row += 1
    else:
        pass
of.close()
print("Finished.")
