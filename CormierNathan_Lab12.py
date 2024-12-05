import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

# REFERENCES:
# CormierNathan_Project3.py - adapted csv reading code for part 1a)
# Numerical Methods For Physics (Python) - Alejandro L. Garcia 2E revised, 2017. pgs 107 to 139

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

# using np.polynomial to generate a polynomial which fits to time and CO2 data
poly_fit = np.polynomial.Polynomial.fit(t_axis,y_axis_CO2,deg=3)

# computing the values of the fit polynomial across all times for plotti
poly_fit_plot = poly_fit(t_axis)

# 1c)
# Produce a 2 panel plot (subplots?) in which the top panel shows the
# original data with the fitted long term trend, and the bottom panel
# shows the residuals (data minus model). Include this plot in your
# submission as CormierNathan_Lab12_Fig1.png

# defining 1 figure with 2 data sets
fig, (ax1,ax2) = plt.subplots(2)

# formatting and labeling for the top panel graph (original data with trendline)
ax1.set_title('Original data with the fitted longterm trendline')
ax1.set_xlabel('Year')
ax1.set_ylabel('CO2 (ppm)')

# plotting the original data with polynomial fit curve overtop over t_axis
ax1.plot(t_axis,y_axis_CO2,'-.0',t_axis,poly_fit_plot,'.') 

# formatting and labeling for the bottom panel graph (residual data)
ax2.set_title('Residuals')
ax2.set_xlabel('Year')
ax2.set_ylabel('Residual CO2 (ppm)')

# plotting the residual data over t_axis 
ax2.plot(t_axis,y_axis_CO2-poly_fit_plot,'-',c='k')

# fixes layout issue, stops plots from having overlapping titles
fig.tight_layout()
plt.show()

#--------------------------------------------------#

# Part 2

# By trial and error, experiment with how well you can fit the periodic variation of the residuals with
# a simple sinusoidal function of the form f(t) = A sin(2pi(t/T) + phi). Note that you have 3 parameters 
# that you can adjust: the amplitude A, the period T, and the phase phi. Does the attempt to fit this
# sinusoid suggest that you need to revisit yout polynomial fit?

# Use functions in numpy.fft or scipy.fftpack to estimate the periods present in your fit residuals.
# Do(es) the period(s) you obtain agree with your trial and error estimate?