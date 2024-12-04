import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

# Part 1

# 1a)
# Read the data from the file and use a method to select data only
# from dates from 1981 to 1990. Plot the measurements to check
# that this selection reproduces figure 5.1 in the text.

# reading the csv file from its raw string path and assigning it to csv_out
csv_out = pd.read_csv(r'C:\Users\natha\Desktop\UWO\2024-2025\1st Semester\Physics 3926 - Computer simulations\Python\co2_mm_mlo.csv',skiprows=52)

# isolating only the needed data into easily manipulable np array
csv_data_all = np.array(csv_out)

# defining the index values to isolate data from 1981 to 1990 (exclusive) 
start_index = 327-54
end_index = 434-53

# by looking the csv, I know I need only the data from rows start_index to end_index for 1981 to 1990 (exclusive)
csv_data_81_90ex = csv_data_all[start_index:end_index,:]

# plotting the monthly average ppm of CO2 from 1981 to 1990
t_axis = csv_data_81_90ex[0:119,2]      # setting the time equal to the decimal dates
y_axis_CO2 = csv_data_81_90ex[0:119,3]  # setting the ppm CO2 to the average measured values from the csv

# reproducing the plot from Figure 5.1 in the textbook (with nicer formatting)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('Average ppm CO2 from 1981 to 1990 (exclusive)')
plt.plot(t_axis,y_axis_CO2,'-.')
plt.show()

# 1b) (SEE PAGE 137)
# Detrend the 9 year dataset by fitting a low order polynomial with
# np.polynomial or sp.signal.detrend, and subtract the fit prediction
# from each data point to obtain the residuals. You need to use your
# judgement to decide what order polynomial to adopt, and a critical
# consideration is that you are going to need to extrapolate your fit.

# 1c)
# Produce a 2 panel plot (subplots?) in which the top panel shows the
# original data with the fitted long term trend, and the bottom panel
# shows the residuals (data minus model). Include this plot in your
# submission as CormierNathan_Lab12_Fig1.png

# REFERNCES:
# CormierNathan_Project3.py - adapted 