import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile
from beam_fitting_code import image
import scipy.optimize as scpo
from scipy.optimize import curve_fit
from matplotlib import gridspec
from pandas import plotting
import imageio
import cv2
import os
import csv

#CONFIGURATION
X_CROP = None
Y_CROP = None 
amplitude_range = np.arange(-0.5,0.5,0.05)
images_in_set = len(amplitude_range)

#create empty arrays for pixel values
x_intensities=[]
y_intensities=[]
x_max_intensities=[]
y_max_intensities=[]
max_pixel=[]
amplitude_corrections_x =[]
amplitude_corrections_y = []

#measurement numbers to analyse
measurements = np.arange(42,44,1)



#END CONFIGURATION

def main(): 
    
    for measure_value in measurements:
    #set path to where images are saved
        path = r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\January\29\Measure {}".format(measure_value)

        #initialise profile class
        prof=profile(path)

        #create empty arrays for pixel values
        x_intensities=[]
        y_intensities=[]
        x_max_intensities=[]
        y_max_intensities=[]
        max_pixel=[]

        #crop it and get beam radius
        z, wx, wy = prof.analyseBeamProfile(path, crop_x=X_CROP, crop_y=Y_CROP)
        
        for x in range(0,(3*len(amplitude_range)+1),1):
            d =  path + "/{}.png". format(x)
            yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = prof.plotSingle(d, crop_x=X_CROP, crop_y=Y_CROP)

        #find the maximum pixel
            pix = imageio.imread(d)
            max_pixel.append(np.amax(pix))
            if np.amax(pix) == 255:
                warnings.warn("Maximum pixel is saturated in Image " + str(x))
            elif np.amax(pix) >= 245:
                warnings.warn("Maximum pixel value close to saturation: Image "+ str(x) + " has max pixel value " + str(np.amax(pix)))
            elif np.amax(pix) <=50:
                warnings.warn("Maximum pixel value is below 50 in Image " + str(x))

            #take out max intensity values
            x_intensities.append(xInt)
            y_intensities.append(yInt)
            x_max_intensities.append(max(xInt))
            y_max_intensities.append(max(yInt))

        #split image values into runs and average 
        wx_1 = wx[1:(images_in_set+1)] 
        wx_2 = wx[(images_in_set+1):2*(images_in_set)+1]
        wx_3 = wx[2*(images_in_set)+1::]
        wx_average = (wx_1 + wx_2 + wx_3)/3
        wx_std = np.std(np.vstack([wx_1, wx_2, wx_3]), axis=0)
        wx_standard_err = wx_std / np.sqrt(3)

        wy_1 = wy[1:(images_in_set+1)]
        wy_2 = wy[(images_in_set+1):2*(images_in_set)+1]
        wy_3 = wy[2*(images_in_set)+1::]
        wy_average = (wy_1 + wy_2 + wy_3)/3
        wy_std = np.std(np.vstack([wy_1, wy_2, wy_3]), axis=0)
        wy_standard_err = wy_std/ np.sqrt(3)
        
        #repeats and averaging intensities
        x_max_intensity_1 = np.array(x_max_intensities[1:(images_in_set+1)])
        x_max_intensity_2 = np.array(x_max_intensities[(images_in_set+1):2*(images_in_set)+1])
        x_max_intensity_3 = np.array(x_max_intensities[2*(images_in_set)+1::])
        x_max_intensity_average = (x_max_intensity_1 + x_max_intensity_2 + x_max_intensity_3)/3

        y_max_intensity_1 = np.array(y_max_intensities[1:(images_in_set+1)])
        y_max_intensity_2 = np.array(y_max_intensities[(images_in_set+1):2*(images_in_set)+1])
        y_max_intensity_3 = np.array(y_max_intensities[2*(images_in_set)+1::])
        y_max_intensity_average = (y_max_intensity_1 + y_max_intensity_2 + y_max_intensity_3)/3

        #take minimum waist values
        print("Minimum beam waist in x = " + str(min(wx_average)))
        print("Minimum beam waist in y = " + str(min(wy_average)))
        z_x = np.argmin(wx_average)
        z_y = np.argmin(wy_average)
        print("Amplitude correction for beam waist in x = " + str(amplitude_range[z_x]))
        print("Amplitude correction for beam waist in y = " + str(amplitude_range[z_y]))

        #create lists of the amplitude correction values
        amplitude_corrections_x.append(amplitude_range[z_x])
        amplitude_corrections_y.append(amplitude_range[z_y])
        print(amplitude_corrections_x)
        print(amplitude_corrections_y)

        #plot error bars of the minimum waist vs milliwave amplitude
        plt.figure()
        plt.errorbar(amplitude_range*1000, wx_average, yerr=wx_standard_err, marker='o', linestyle='', label='Mean beam waist in x', color='blue')
        plt.errorbar(amplitude_range*1000, wy_average, yerr=wy_standard_err, marker='o', linestyle='',  label= 'Mean beam waist in y', color='orange')
        plt.ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
        plt.xlabel('Amplitude / Milliwaves')
        plt.legend()

    plt.show()

if __name__ == "__main__":
    main()