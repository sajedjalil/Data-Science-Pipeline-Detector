import pandas
import os
import matplotlib.pyplot as plt
import numpy

def trace_converter(value):
    if value.strip() == "T":
        return 0.005
    elif value.strip() == "M":
        return None
    return float(value)

converters = {
    "PrecipTotal": trace_converter,
    "SnowFall": trace_converter
}

weather_data = pandas.read_csv("../input/weather.csv", header=0, parse_dates=[1],  na_values=["M", "-"], converters=converters)
weather_data.loc[weather_data.Tavg.isnull(), "Tavg"] = (weather_data.Tmin + weather_data.Tmax) / 2.
weather_data["Year"] = weather_data.Date.dt.year

weather_1 = weather_data[weather_data.Station == 1].drop("Station", axis=1)
weather_2 = weather_data[weather_data.Station == 2].drop("Station", axis=1)

weather_joined = weather_1.merge(weather_2, on="Date", suffixes=("_1", "_2"), sort=True)


weather_by_ys = weather_joined.pivot_table(index=["Year_1"], values=["PrecipTotal_1", "PrecipTotal_2"], aggfunc=sum)
weather_by_ys.plot(kind="bar", title="PrecipTotal by Year and Station")
plt.savefig("precip.png")

weather_by_ys.to_csv("precip.csv")

plt.clf()
(weather_joined.PrecipTotal_1 - weather_joined.PrecipTotal_2).hist(bins=30)
plt.savefig("precip_diff_hist.png")