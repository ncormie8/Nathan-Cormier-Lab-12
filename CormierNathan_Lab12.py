import numpy as np
import scipy as sp
import pandas as pd

# Part 1

# 1a)
# Read the data from the file and use a method to select data only
# from dates between 1981 and 1990. Plot the measurements to check
# that this selection reproduces figure 5.1 in the text.


# reading the csv file from its raw string path and assigning it to csv_out
csv_out = pd.read_csv(r'C:\Users\natha\Desktop\UWO\2024-2025\1st Semester\Physics 3926 - Computer simulations\Python\co2_mm_mlo.csv',skiprows=57)

# isolating only the needed data into easily manipulable np array
csv_data_all = np.array(csv_out)

# by reading the csv, I know I need only the data from rows 327 to 446 for 1981 to 1990
csv_data_81_90 = csv_data_all[327-57:446-57,:]
print(csv_data_81_90)

# 1b)
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