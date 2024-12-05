import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime , timedelta

# REFERENCES:
# CormierNathan_Project3.py - adapted csv reading code for part 1a)
# lecture12b_Ftdemo.py - adapted frequency determination code for part 2
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

# defining the index values to isolate data from [1981:1990) (exclusive) 
start_index = 327-54
end_index = 434-53

# by looking the csv, I know I need only the data from rows start_index to end_index for 1981 to 1990 (exclusive)
csv_data_81_90ex = csv_data_all[start_index:end_index,:]

# plotting the monthly average ppm of CO2 from 1981 to 1990
t_axis = csv_data_81_90ex[:,2]      # setting the time equal to the decimal dates
y_axis_CO2 = csv_data_81_90ex[:,3]  # setting the ppm CO2 to the average measured values from the csv

# reproducing the plot from Figure 5.1 in the textbook (with nicer formatting)
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('Average ppm CO2 from 1981 to 1990 (exclusive)')
plt.plot(t_axis,y_axis_CO2,'-',c='k')
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

# defining residuals y axis values
residuals = y_axis_CO2-poly_fit_plot

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
ax2.plot(t_axis,residuals,'-',c='k')

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

# defining fucntion sinusoid to generate a sinusoidal fit curve of the form given in the problem
def sinusoid(t, args):
    '''Generates array of values f corresponding to the value of f at time t. f(t) has the form
    A*np.sin(2*np.pi*(t/T) + phi) + y_offset. Intakes parameters t as the array of all times which f(t) should
    be calculated, and args() as an array containing the amplitude, period, phase shift, and y offset to be
    used when calculating f(t). Returns array containing all values of f at all times t.'''
    A = args[0]         # ampltiude set to first input argument
    T = args[1]         # period set to second input argument
    phi = args[2]       # phase shift set to third input argument
    y_offset = args[3]  # y offset set to the fourth input argument

    # initializing zeros sin function value array
    f = np.zeros(np.size(t))

    # calculating all of the values of the sin function over the input time range      
    f = A*np.sin(2*np.pi*(t/T) + phi) + y_offset
    
    return f

# TRIAL AND ERROR SECTION
# trial and error fit curve variable declaration
amp = 3
est_period = 1
phase_shift = -np.pi/8
y_offset = 0
args_tANDe = [amp,est_period,phase_shift,y_offset]

# generating my trial and error sinusoidal curve of best fit for residuals over times t_axis
sin_fit_tANDe = sinusoid(t_axis,args_tANDe)

# plotting residuals with overlayed estimated sinusoidal fit curve
plt.plot(t_axis,residuals,'-',t_axis,sin_fit_tANDe,'-.')
plt.title('Residual CO2 data with overlayed estimated sinusoidal fit curve')
plt.xlabel('Year')
plt.ylabel('Residual CO2 (ppm)')
plt.legend(['Residual CO2 data','Estimated sin fit curve'])
plt.show()

# Since the trial and error sinusoid fits the residuals plot quite well, there is no indication
# that I need to revisit my polynomial fit.

# np.fft SECTION
spacing = t_axis[1]-t_axis[0]

# performing fft analysis to determine new period for sinusoidal fit curve
res_t = np.fft.fft(residuals)
freqs = np.fft.fftfreq(np.size(t_axis),spacing)
res_t_positive = res_t[:np.size(t_axis)//2]
freqs_positive = freqs[:np.size(t_axis)//2]

# finding the dominant frequency from FFT
biggest_freq= np.argmax(np.abs(res_t_positive))
new_freq = freqs[biggest_freq]

# calculating the new period from the new frequency
new_T = 1/new_freq

# Yes, the period obtained using fft analysis aligns well with the estimated value.
# The estimated period was 1, and the calcuated value is 0.9996000000001003.

#--------------------------------------------------#

# Part 3

# Combine your polynomial and sinusoidal fits on the 1981-1990 to propose a model for
# the time variation of the CO2 concentration. How well does you model fit the data outside
# of the 1981-1990 window? Use your model to predict when the atmospheric concentration
# first reached 400 ppm and compare this with the actual date when it happened. Make a
# plot of your model and data and include this plot in your submission as 
# LastnameFirstname_Lab12_Fig2.png.

# defining parameters for times and CO2 concetrations for given data points from the csv
t_axisall = csv_data_all[:,2]      
y_axis_CO2all = csv_data_all[:,3]

# determining a new equation for a polynomial fit for all csv data
poly_fit_all = np.polynomial.Polynomial.fit(t_axisall,y_axis_CO2all,deg=3)

# determining new polynomial and sinusoidal fits to use for all times
poly_fit_all_use = poly_fit_all(t_axisall)
sin_fit_all = sinusoid(t_axisall,args_tANDe)

# adding the polynomial and sinusoidal fits together to create a complete model of CO2
# concentration changes overtime
complete_model = sin_fit_all + poly_fit_all_use

# plotting code to generate plot of modeled CO2 concentrations over time
plt.plot(t_axisall,complete_model,'-',c='k')
plt.xlabel('Year')
plt.ylabel('Projected concentration of CO2 (ppm)')
plt.title('Proposed time-model of atmospheric CO2 concentration (ppm)')
plt.show()
    
# finding the point in time where the projected CO2 concentration is closest to 400 ppm
diffs_model = np.abs(complete_model-400)
t400model = t_axisall[np.argmin(diffs_model)]
CO2_400model = complete_model[np.argmin(diffs_model)]

# finding the row where the projected time to reach 400 ppm occurs to extract the date
csv_row_model = np.argwhere(csv_data_all[:,2]==t400model)
csv_row_m = csv_row_model[0,0]

# determining the year, month, and day the the model projects 400 ppm will be reached in a MM/DD/YYYY format
year_m = int(csv_data_all[csv_row_m,0])
month_m = int(csv_data_all[csv_row_m,1])
day_m = int((t400model - year_m)*365)
model_400_MMDDYY = str(month_m) + '/' + str(day_m) + '/' + str(year_m)
print('On '+model_400_MMDDYY+' the projected concentration of CO2 reaches '+str(CO2_400model)+' ppm.')

# finding the point in time where the actual CO2 concentration is closest to 400 ppm
diffs_data = np.abs(y_axis_CO2all-400)
t400data = t_axisall[np.argmin(diffs_data)]
CO2_400data = y_axis_CO2all[np.argmin(diffs_data)]

# finding the row where the projected time to reach 400 ppm occurs to extract the date
csv_row_data = np.argwhere(csv_data_all[:,2]==t400data)
csv_row_d = csv_row_data[0,0]

# determining the year, month, and day the the model projects 400 ppm will be reached in a MM/DD/YYYY format
year_d = int(csv_data_all[csv_row_d,0])
month_d = int(csv_data_all[csv_row_d,1])
day_d = int((t400data - year_d)*365)
data_400_MMDDYY = str(month_d) + '/' + str(day_d) + '/' + str(year_d)
print('On '+data_400_MMDDYY+' the actual concentration of CO2 reached '+str(CO2_400data)+' ppm.')