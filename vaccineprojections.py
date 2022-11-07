# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:32:35 2021

@author: Amanda

This program imports data about Covid-19 vaccinations in Germany, and 
calculates the rate of vaccination to predict when certain vaccination
milestones will be met.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
from sklearn.metrics import mean_squared_error


#-------Collect and Store Data-----------------------------------------

#save population data as constants
#data source: https://www.destatis.de/EN/Themes/Society-Environment/Population/Current-Population/Tables/lrbev01.html
pop_total = 83166711
pop_over80 = pop_total * 0.068
pop_over60 = pop_over80 + (pop_total * 0.217)
    
#import vaccination data from .tsv file and store in dataframe
#data source: https://impfdashboard.de/
url = "https://impfdashboard.de/static/data/germany_vaccinations_timeseries_v2.tsv"
raw_data = pd.read_csv(url, sep='\t', parse_dates=["date"])
data = raw_data[["date", "personen_erst_kumulativ", "personen_voll_kumulativ", "indikation_alter_erst", "dosen_biontech_kumulativ", "dosen_moderna_kumulativ", "dosen_astrazeneca_kumulativ"]]
#print(raw_data.columns)

most_recent_date = data["date"].iloc[len(data)-1]
most_recent_cumulative = data["personen_erst_kumulativ"].iloc[len(data)-1]


#-------Perform Trend Line Fitting Calculations------------------------

#linear fit for vaccine data
vax_array_calc = np.array(data[["personen_erst_kumulativ"]])
x = np.arange(1, len(vax_array_calc)+1)
m, b = np.polyfit(x,vax_array_calc,1)

single_day_change = vax_array_calc[len(vax_array_calc)-1] - vax_array_calc[len(vax_array_calc)-2]

#exponential fit for vaccine data
a, z = np.polyfit(x, np.log(vax_array_calc),1)

#2nd degree polynomial fit for vaccine data
c,d,e = np.polyfit(x, vax_array_calc, 2)


#-------Store Trend Line Data Points in Dataframe-----------------------

#extend the dataframe to do calculations for future vaccination trends
current_date = dt.date.today()
future_dates = pd.DataFrame({"date": pd.date_range(start=current_date, end="2021-12-31")})
data = data.append(future_dates)

#index numbers for doing calculations
index = np.arange(1,len(data)+1)

#add additional columns
data["vaccine deliveries"] = 0
data["index"] = index
data["linear fit - current rate"] = m * data["index"] + b
data["exponential fit"] = np.exp(z) * np.exp(a*data["index"])
data["degree2 fit"] = c * (data["index"] ** 2) + d * data["index"] + e


#-------Calculate and Display Projected Milestone Dates-----------------------------

#calculate and display projected milestone dates
over60_date = int((pop_over60 - b) / m)
over80_date = int((pop_over80 - b) / m)
total_date = int((pop_total - b) / m)
target_rate = int((pop_total - b) / len(index))
#print("Most recent vaccination data is from:", most_recent_date.strftime("%Y-%m-%d"))
print("On the last day for which data is available,", int(single_day_change), "people received their first vaccination.")
print("A total of", most_recent_cumulative, "people have received their first vaccination since 2020-12-27.")
#print("Vaccination Rate (linear fit model):", int(m), "per day.")
#print("All over-80s will have at least one dose by", current_date + dt.timedelta(days=over80_date), ".")
#print("All over-60s will have at least one dose by", current_date + dt.timedelta(days=over60_date), ".")
#print("The entire population will have at least one dose by", current_date + dt.timedelta(days=total_date), ".")
#print("To vaccinate everyone by 2021-12-31, a vaccination rate of at least", target_rate, "per day is needed.")


#-------Format Data for Plotting---------------------------------------------------

#remove NaNs
data["personen_erst_kumulativ"] = data["personen_erst_kumulativ"].replace(np.nan, -pop_total)

#convert data from dataframe columns to numpy arrays
all_dates = np.array(data[["date"]])
pop_total_array = np.full(len(all_dates), pop_total)
pop_over80_array = np.full(len(all_dates), pop_over80)
pop_over60_array = np.full(len(all_dates), pop_over60)
lfc_array = np.array(data[["linear fit - current rate"]])
exp_array = np.array(data[["exponential fit"]])
vax_array_plot = np.array(data[["personen_erst_kumulativ"]])
vax2_array_plot = np.array(data[["personen_voll_kumulativ"]])
deg2_array = np.array(data[["degree2 fit"]])
age_array = np.array(data[["indikation_alter_erst"]])

biontech = np.array(data[["dosen_biontech_kumulativ"]])
moderna = np.array(data[["dosen_moderna_kumulativ"]])
az = np.array(data[["dosen_astrazeneca_kumulativ"]])

#-------Calculate errors for each trend line---------------------------

lin_err = mean_squared_error(vax_array_calc, lfc_array[:len(vax_array_calc)], squared=False)
exp_err = mean_squared_error(vax_array_calc, exp_array[:len(vax_array_calc)], squared=False)
deg2_err = mean_squared_error(vax_array_calc, deg2_array[:len(vax_array_calc)], squared=False)

#print("Linear Error: ", lin_err)
#print("Exponential Error: ", exp_err)
#print("Polynomial Error: ", deg2_err)
#print(lin_err - deg2_err)


#-------Select the best model and display data from that model--------------------

if min(lin_err, exp_err, deg2_err) == lin_err:
    print("The linear fit model has the lowest error with the current data.")
    lin_weight = 3
    lin_label = "BEST FIT: Linear Fit (Ax + B)"
    exp_weight = 1
    exp_label = "Exponential Fit (e^(Ax))"
    deg2_weight = 1
    deg2_label = "Polynomial Fit (Ax^2 + Bx + C)"
elif min(lin_err, exp_err, deg2_err) == exp_err:
    print("The exponential fit model has the lowest error with the current data.")
    lin_weight = 1
    lin_label = "Linear Fit (Ax + B)"
    exp_weight = 3
    exp_label = "BEST FIT: Exponential Fit (e^(Ax))"
    deg2_weight = 1
    deg2_label = "Polynomial Fit (Ax^2 + Bx + C)"
elif min(lin_err, exp_err, deg2_err) == deg2_err:
#    print("The polynomial fit model has the lowest error with the current data.")
    #calculate and display milestone data using this model
 #   deg2_80date = all_dates[np.argwhere(deg2_array >= pop_over80)[0,0]]
#    deg2_60date = all_dates[np.argwhere(deg2_array >= pop_over60)[0,0]]
    done = all_dates[np.argwhere(deg2_array >= pop_total)[0,0]]
    doneStr = str(np.datetime_as_string(done, unit="D"))[2:12]
#    doneStrShort = doneStr[2:12]
#    print("Using this model, all first doses will be done by: ", doneStr)
    #adjust variables for plotting the data, to emphasize the best fit line
    lin_weight = 1
    lin_label = "Linear Fit (Ax + B)"
    exp_weight = 1
    exp_label = "Exponential Fit (e^(Ax))"
    deg2_weight = 3
    deg2_label = "BEST FIT: Polynomial Fit (Ax^2 + Bx + C)"


#-------Plot data---------------------------------------------------------------------

#set up figure
dateFormat_past = mdates.DateFormatter("%b-%d-%y")
dateFormat_future = mdates.DateFormatter("%b %Y")
fig, axs = plt.subplots(1,2, figsize=(15,5))
fig.suptitle("Covid-19 Vaccinations in Germany\n(data from " + most_recent_date.strftime("%Y-%m-%d") + ", accessed on " + str(current_date) + ")", fontsize=16, y=1.05)

max_yval = max(exp_array[len(vax_array_calc)-1], lfc_array[len(vax_array_calc)-1], deg2_array[len(vax_array_calc)-1])

#on axs[0], plot trendlines with existing data
plt.setp(axs[0].get_xticklabels(), ha="right", rotation=45)
axs[0].scatter(all_dates[:len(vax_array_calc)], vax_array_plot[:len(vax_array_calc)], label="Actual Vaccinations (first dose)", c="royalblue", zorder=4)
#axs[0].scatter(all_dates[:len(vax_array_calc)], vax2_array_plot[:len(vax_array_calc)], label="Actual Vaccinations (second dose)", c="green", zorder=6)
# axs[0].scatter(all_dates[:len(vax_array_calc)], age_array[:len(vax_array_calc)], label="First dose given due to age", c="royalblue", zorder=5, alpha=0.5)
#axs[0].vlines("2021-03-01", ymin=0, ymax=max_yval, label="3/1/21: Under-65s can start getting the AstraZeneca vaccine.", color='green')
#axs[0].vlines("2021-03-15", ymin=0, ymax=max_yval, label="3/15/21: Germany suspends use of AZ vaccine due to blood clot concerns.", color='darkorange')
axs[0].plot(all_dates[:len(vax_array_calc)], exp_array[:len(vax_array_calc)], label=exp_label, c="red", linewidth=exp_weight, zorder=1)
axs[0].plot(all_dates[:len(vax_array_calc)], lfc_array[:len(vax_array_calc)], label=lin_label, c="darkorchid", linewidth=lin_weight, zorder=2)
axs[0].plot(all_dates[:len(vax_array_calc)], deg2_array[:len(vax_array_calc)], label=deg2_label, c="deeppink", linewidth=deg2_weight, zorder=3)
axs[0].set_title("Past Vaccination Data")
axs[0].xaxis.set_major_formatter(dateFormat_past)
axs[0].set_ylabel("Population (in millions)")

#max_yval = max(exp_array[len(vax_array_calc)-1], lfc_array[len(vax_array_calc)-1], deg2_array[len(vax_array_calc)-1])
yticks = np.arange(0, max_yval*1.2, max_yval/8)
ylabels = np.zeros(len(yticks))
for i in range(0, len(yticks)):
    ylabels[i] = round(yticks[i]/1000000, 2)

axs[0].set_yticks(yticks)
axs[0].set_yticklabels(ylabels)
axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
axs[0].grid(True)

#on axs[1], plot trendlines with existing data and include future projections to the end of the year
axs[1].set_title("Future Vaccination Projections")
axs[1].scatter(all_dates, vax_array_plot, label="Actual Vaccinations", c="royalblue", zorder=4)
axs[1].plot(all_dates, pop_total_array, label="Total Population", c="mediumturquoise", ls=":", linewidth=3)
axs[1].plot(all_dates, pop_over80_array, label="Population over 80", c="green", ls=":", linewidth=3)
axs[1].plot(all_dates, pop_over60_array, label="Population over 60", c="darkorange", ls=":", linewidth=3)
axs[1].plot(all_dates, exp_array, label=exp_label, c="red", linewidth=exp_weight)
axs[1].plot(all_dates, lfc_array, label=lin_label, c="darkorchid", linewidth=lin_weight)
axs[1].plot(all_dates, deg2_array, label=deg2_label, c="deeppink", linewidth=deg2_weight)
axs[1].xaxis.set_major_formatter(dateFormat_future)
axs[1].set_ylabel("Population (in millions)")
axs[1].set_ylim(-2000000, pop_total+5000000)
axs[1].set_yticks([0, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000])
axs[1].set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80])
axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.55))
axs[1].annotate(doneStr + ": 1st doses done", (done,deg2_array[np.argwhere(deg2_array >= pop_total)[0,0]]), xytext=(done,deg2_array[np.argwhere(deg2_array >= pop_total)[0,0]]-7500000), bbox=dict(boxstyle="round", fc="0.8"))
axs[1].grid(True)


#datelabels = all_dates[:len(vax_array_calc)]
#biontech_short = biontech[:len(vax_array_calc)]
#print(datelabels)
#print(len(datelabels))
#print(np.shape(datelabels))
#print(len(biontech_short))
#print(np.shape(biontech_short))
#print(biontech_short)

#axs[2].set_title("Doses by Manufacturer")
#axs[2].bar(datelabels, biontech_short, label='BioNTech', linewidth=0)


plt.show()
