import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import pandas as pd

#Read our data into dataframe
df = pd.read_csv("../input/train.csv")

#Prepare the dataframe for grouping on date
df["datetime"] = pd.to_datetime(df["datetime"])
df['datetime_minus_time'] = df["datetime"].apply(lambda df: datetime.datetime(year=df.year, month=df.month, day=df.day))
df.set_index(df["datetime_minus_time"],inplace=True)

#Get our data
avg_counts_per_month = df["count"].resample("M", how="sum")

xticks = [k.strftime('%b %Y') for k in list(avg_counts_per_month.index)]

#plotting
plt.barh([x for x in range(len(xticks))], list(avg_counts_per_month), align='center', alpha=0.4)
plt.yticks([x for x in range(len(xticks))], xticks,fontsize=9)

plt.xlabel("Count", fontsize=14)
plt.title("The count of rentals per month shows signs of growth and summer-time peaks", fontsize=11, fontweight='bold')
plt.gca().xaxis.grid(True,linestyle='-',alpha=0.1)
plt.gca().yaxis.grid(True,alpha=0.4)
plt.savefig("monthly-rental-counts.png")