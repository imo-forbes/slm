# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:35:37 2021

@author: imoge
"""

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


#file images are being taken from
d = "./images/2021/November/29/Measure 5"
amplitude_range= np.arange(-1,1,0.1)

profile=profile(d)

#collect beam radius arrays from images
z, wx, wy = profile.analyseBeamProfile(d)

#collect intensity values from images
x_intensities=[]
y_intensities=[]
x_max_intensities=[]
y_max_intensities=[]
max_pixel=[]


for x in range(0,(3*len(amplitude_range)+1),1):
    d =  r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\29\Measure 5" + "/{}.png". format(x)
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle(d)

    #find the maximum pixel
    pix = imageio.imread(d)
    max_pixel.append(np.amax(pix))
    if np.amax(pix) == 255:
        warnings.warn("Maximum pixel is saturated in Image " + str(x))
    elif np.amax(pix) >= 225:
        warnings.warn("Maximum pixel value close to saturation: Image "+ str(x) + " has max pixel value " + str(np.amax(pix)))
    elif np.amax(pix) <=50:
        warnings.warn("Maximum pixel value is below 50 in Image " + str(x))

    #take out max intensity values
    x_intensities.append(xInt)
    y_intensities.append(yInt)
    x_max_intensities.append(max(xInt))
    y_max_intensities.append(max(yInt))

#remove background - code makes image 0 the background image
wx_corrected = []
wy_corrected = []
for image_no in range(1,len(wx),1):
    wx_corrected.append(wx[image_no] - wx[0])
    wy_corrected.append(wy[image_no] - wy[0])
    
    
#breaks list of values into repeats and averages
images_in_set = len(amplitude_range)
print(images_in_set)

#repeats and averaging beam radius
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

#repeats and averaging intensities
x_max_intensity_1 = np.array(x_max_intensities[1:(images_in_set+1)])
x_max_intensity_2 = np.array(x_max_intensities[(images_in_set+1):2*(images_in_set)+1])
x_max_intenisty_3 = np.array(x_max_intensities[2*(images_in_set)+1::])
x_max_intensity_average = (x_max_intensity_1 + x_max_intensity_2 + x_max_intenisty_3)/3

y_max_intensity_1 = np.array(y_max_intensities[1:(images_in_set+1)])
y_max_intensity_2 = np.array(y_max_intensities[(images_in_set+1):2*(images_in_set)+1])
y_max_intenisty_3 = np.array(y_max_intensities[2*(images_in_set)+1::])
y_max_intensity_average = (y_max_intensity_1 + y_max_intensity_2 + y_max_intenisty_3)/3


#error analysis formulas

def gaus(x,a,x0,sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+offset

def mean(x,y):
    return sum(x*y)/len(x)

def sigma(x,y):
    return math.sqrt(sum(y*(x-mean(x,y))**2)/len(x))

#finding optimum values for gaussain fit    

hints = [min(wx_average),mean(amplitude_range * 1000, wx_average),sigma(amplitude_range*1000,wx_average), 0]

print("Hints", hints)
popt_x,pcov = curve_fit(gaus,
    amplitude_range*1000,
    wx_average,
    sigma=wx_standard_err,
    p0=hints,
    absolute_sigma=True,
    maxfev=1000000)
#popt_y,pcov = curve_fit(gaus,amplitude_range*1000,wy_average,p0=[min(wy_average),mean(amplitude_range*1000,wy_average),sigma(amplitude_range*1000,wy_average)], maxfev=10000)

print("ACtual", popt_x)

err_x = pcov[1,1]**(1/2) 
print(pcov)
print("Err x", err_x)
#plot of beam waist vs milliwaves
fig = plt.figure(figsize=(9,12))
gs = gridspec.GridSpec(7, 6, hspace=0,wspace=0)

main_plot_ax = fig .add_subplot(gs[:-2, :-1])
x_res_ax = fig.add_subplot(gs[-2, :-1])
#y_res_ax = fig.add_subplot(gs[-1, :-1])
x_hist_ax = fig.add_subplot(gs[-2, -1])
#y_hist_ax = fig.add_subplot(gs[-1, -1])

x_hist_ax.get_yaxis().set_visible(False)
#y_hist_ax.get_yaxis().set_visible(False)
x_hist_ax.get_xaxis().set_visible(False)
#y_hist_ax.get_xaxis().set_visible(False)
residual_y_lim = (-4, 3)

x_res_ax.set_ylim(residual_y_lim)
#y_res_ax.set_ylim(residual_y_lim)
x_res_ax.set_yticks([-10, 0,10])
#y_res_ax.set_yticks([-5, 0,5])

#fitting gaussians
main_plot_ax.plot(amplitude_range*1000,gaus(amplitude_range*1000,*popt_x),label='Gaussian Fit for beam waist in x')
#main_plot_ax.plot(amplitude_range*1000,gaus(amplitude_range*1000,*popt_y),label='Gaussian Fit for beam waist in y')
main_plot_ax.errorbar(amplitude_range*1000, wx_average, yerr=wx_standard_err, marker='o', linestyle='', label='Mean beam waist in x', color='blue')
#main_plot_ax.errorbar(amplitude_range*1000, wy_average, yerr=standard_error(wy_average), marker='o', linestyle='',  label= 'Mean beam waist in y', color='orange')
main_plot_ax.set_ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
main_plot_ax.set_xlabel('Amplitude / Milliwaves')
main_plot_ax.legend()


#normalised residuals
x_norm_residuals=[]
y_norm_residuals=[]

x_norm_residuals = (wx_average - gaus(amplitude_range*1000,*popt_x))/wx_standard_err
#y_norm_residuals = (wy_average - gaus(amplitude_range*1000,*popt_y))/standard_error(wy_average)

x_res_ax.scatter(amplitude_range*1000, x_norm_residuals, color='blue', marker='x')
#x_res_ax.stem(amplitude_range*1000, x_norm_residuals, linefmt=None, markerfmt='x',basefmt="k")
#y_res_ax.scatter(amplitude_range*1000, y_norm_residuals, color='orange', marker='x')
#y_res_ax.stem(amplitude_range*1000, y_norm_residuals, linefmt='C1', markerfmt='C1x', basefmt='k')
x_res_ax.axhline(y=0, c='k' )
#y_res_ax.axhline(y=0, c='k')
x_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='green', alpha=0.25)
#y_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='green', alpha=0.25)
#y_res_ax.set_xlabel('Milliwaves')
x_res_ax.set_xlabel('Amplitude / Milliwaves')



plt.text(-0.08, 0.5, "Norm. Residuals", verticalalignment="center", horizontalalignment="center", transform=x_res_ax.transAxes, rotation=90, fontsize=10)

#plot histogram
x_hist_ax.hist(x_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='blue')
#y_hist_ax.hist(y_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='orange')

#finding the minimum beam waist from the fit
# dots_x,dots_y,dots_y_err = amplitude_range, wx_average, yerr


wx_average = gaus(amplitude_range*1000,*popt_x)
print(min(wx_average))
z_x = np.where(wx_average== min(wx_average))
print(z_x)
print(amplitude_range[z_x]*1000)
#print(standard_error(wx_average))
#print(standard_error(wx_average))
#print(standard_error(wy_average))

plt.figure()
plt.scatter(range(0, len(amplitude_range)),max_pixel[0:len(amplitude_range)])
plt.xlabel('Image Number')
plt.ylabel('Value of Maximum Pixel in Image')

#plot of beam waist vs milliwaves
# plt.figure(figsize=(10,4))
# plt.plot(amplitude_range*1000,gaus(amplitude_range,*popt_x),label='Gaussian Fit')
# plt.plot(amplitude_range*1000,gaus(amplitude_range,*popt_y),label='Gaussian Fit')
# plt.errorbar(amplitude_range*1000, wx_average, yerr=standard_error(wx_average), marker='o', linestyle='', label='Mean beam waist in x', color='blue')
# plt.errorbar(amplitude_range*1000, wy_average, yerr=standard_error(wy_average), marker='o', linestyle='',  label= 'Mean beam waist in y', color='orange')
# plt.ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
# plt.xlabel('Milliwaves ')
# plt.legend()


# #plot of maximum intensity vs amplitude
# plt.figure(figsize=(10,4))
# plt.plot(amplitude_range,gaus(amplitude_range,*popt_x),label='Gaussian Fit')
# plt.plot(amplitude_range,gaus(amplitude_range,*popt_y),label='Gaussian Fit')
# plt.errorbar(amplitude_range, x_max_intensity_average, yerr=standard_error(x_max_intensity_average), marker='o', linestyle='', label='Peak Intensity in x', color='blue')
# plt.errorbar(amplitude_range , y_max_intensity_average, yerr=standard_error(y_max_intensity_average), marker='o', linestyle='',  label= 'Peak Intensity in y', color='orange')
# plt.ylabel('Intensity /')
# plt.xlabel('Amplitude /')
# plt.legend()

#plot of maximum intensity vs amplitude
# plt.figure(figsize=(9,8))
# plt.errorbar(amplitude_range*1000, x_max_intensity_average, yerr=standard_error(x_max_intensity_average), marker='o', linestyle='', label='Peak Intensity in x', color='blue')
# plt.errorbar(amplitude_range*1000, y_max_intensity_average, yerr=standard_error(y_max_intensity_average), marker='o', linestyle='',  label= 'Peak Intensity in y', color='orange')
# plt.ylabel('Intensity / Wm$^{-2}$')
# plt.xlabel('Milliwaves')
# plt.legend()

# print(max(x_max_intensity_average))
# print(max(y_max_intensity_average))
# z_x = np.where(x_max_intensity_average== max(x_max_intensity_average))
# z_y = np.where(y_max_intensity_average ==max(y_max_intensity_average))
# print(z_x)
# print(z_y)
# print(amplitude_range[35])
# print(amplitude_range[36])

plt.show()