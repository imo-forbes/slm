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
import cv2
import os
import matplotlib

amplitude_range= np.arange(-1, 1, 0.05) 
print(amplitude_range)

#file images are being taken from
range_to_fit_start = 10
range_to_fit_end = 35
path = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\December\18\Measure 48"
profile=profile(path)

#collect beam radius arrays from images
z, wx, wy = profile.analyseBeamProfile(path, crop_x=100, crop_y=100)

# plt.scatter(range(1,4,1), wx[1::])
# plt.scatter(range(1,4,1), wy[1::])
# plt.show()
# wx_average = (wx[1] + wx[2] + wx[3])/3
# wy_average = (wy[1] + wy[2] + wy[3])/3
# print(wx_average)
# wx_std = np.std(wx[1::])
# wx_standard_err = wx_std / np.sqrt(3)
# print(wx_standard_err)
# wy_std = np.std(wy[1::])
# wy_standard_err = wy_std / np.sqrt(3)
# print(wy_standard_err)
# print(wy_average)

#collect intensity values from images
x_intensities=[]
y_intensities=[]
x_max_intensities=[]
y_max_intensities=[]
max_pixel=[]

for x in range(0,(4),1):
    d =  path + "/{}.png". format(x)
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle(d, crop_x=100, crop_y=100)

    #take out max intensity values
    x_intensities.append(xInt)
    y_intensities.append(yInt)
    x_max_intensities.append(max(xInt))
    y_max_intensities.append(max(yInt))



# plt.scatter(range(1,4,1), x_max_intensities[1::])
# plt.scatter(range(1,4,1), y_max_intensities[1::])
# plt.show()
# x_max_intensity_average = np.mean(x_max_intensities)
# y_max_intensity_average = np.mean(y_max_intensities)
# print(x_max_intensity_average)
# x_max_intensity_std = np.std(x_max_intensities[1::])
# x_max_intensity_err = x_max_intensity_std / np.sqrt(3)
# print(x_max_intensity_err)
# print(y_max_intensity_average)
# y_max_intensity_std = np.std(y_max_intensities[1::])
# y_max_intensity_err = y_max_intensity_std / np.sqrt(3)
# print(y_max_intensity_err)


#collect intensity values from images
x_intensities=[]
y_intensities=[]
x_max_intensities=[]
y_max_intensities=[]
max_pixel=[]

for x in range(0,(3*len(amplitude_range)+1),1):
    d =  path + "/{}.png". format(x)
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle(d, crop_x=100, crop_y=100)

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
print(np.mean(wx_standard_err))
wx_err_plot=[]
for std_err in wx_standard_err:
    wx_err_in_quad = np.sqrt((std_err)**2 + (3.44e-05)**2)
    wx_err_plot.append(wx_err_in_quad)

print('Difference', np.max(wx_average) - np.min(wx_average))


wy_1 = wy[1:(images_in_set+1)]
wy_2 = wy[(images_in_set+1):2*(images_in_set)+1]
wy_3 = wy[2*(images_in_set)+1::]
wy_average = (wy_1 + wy_2 + wy_3)/3
wy_std = np.std(np.vstack([wy_1, wy_2, wy_3]), axis=0)
wy_standard_err = wy_std/ np.sqrt(3)
print(np.mean(wy_standard_err))
#use errors in quadrature
wy_err_plot=[]
for std_err in wy_standard_err:
    wy_err_in_quad = np.sqrt((std_err)**2 + (9.48e-05)**2)
    wy_err_plot.append(wy_err_in_quad)

print('Difference', np.max(wy_average) - np.min(wy_average))

#repeats and averaging intensities
x_max_intensity_1 = np.array(x_max_intensities[1:(images_in_set+1)])
x_max_intensity_2 = np.array(x_max_intensities[(images_in_set+1):2*(images_in_set)+1])
x_max_intensity_3 = np.array(x_max_intensities[2*(images_in_set)+1::])
x_max_intensity_average = (x_max_intensity_1 + x_max_intensity_2 + x_max_intensity_3)/3
x_intensity_std = np.std(np.vstack([x_max_intensity_1, x_max_intensity_2, x_max_intensity_3]), axis=0)
x_intensity_standard_err = x_intensity_std / np.sqrt(3)

y_max_intensity_1 = np.array(y_max_intensities[1:(images_in_set+1)])
y_max_intensity_2 = np.array(y_max_intensities[(images_in_set+1):2*(images_in_set)+1])
y_max_intensity_3 = np.array(y_max_intensities[2*(images_in_set)+1::])
y_max_intensity_average = (y_max_intensity_1 + y_max_intensity_2 + y_max_intensity_3)/3
y_intensity_std = np.std(np.vstack([y_max_intensity_1, y_max_intensity_2, y_max_intensity_3]), axis=0)
y_intensity_standard_err = y_intensity_std / np.sqrt(3)


#error analysis formulas

def gaus(x,a,x0,sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+offset

def mean(x,y):
    return sum(x*y)/len(x)

def sigma(x,y):
    return math.sqrt(sum(y*(x-mean(x,y))**2)/len(x))

#finding optimum values for gaussain fit    

hints = [0.02,mean(amplitude_range * 1000, wx_average),sigma(amplitude_range*1000,wx_average), 150]

print("Hints", hints)
popt_x,pcov = curve_fit(gaus,
    amplitude_range*1000,
    wx_average,
    sigma=wx_standard_err,
    p0=hints,
    absolute_sigma=True,
    maxfev=1000000)
popt_y,pcov = curve_fit(gaus,
    amplitude_range*1000,
    wy_average,
    p0=hints,
    sigma=wy_standard_err,
    absolute_sigma = True,
    maxfev=100000)

print("ACtual", popt_x)

y = np.poly1d(np.polyfit(amplitude_range[range_to_fit_start::]*1000, wx_average[range_to_fit_start::], 2))
y_y = np.poly1d(np.polyfit(amplitude_range[range_to_fit_start::]*1000, wy_average[range_to_fit_start::], 2))

err_x = pcov[1,1]**(1/2) 
print(pcov)
print("Err x", err_x)
#plot of beam waist vs milliwaves
fig = plt.figure(figsize=(9,8))
gs = gridspec.GridSpec(7, 6, hspace=0,wspace=0)


#font settings for plots

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

main_plot_ax = fig .add_subplot(gs[:-2, :-1])
x_res_ax = fig.add_subplot(gs[-2, :-1])
y_res_ax = fig.add_subplot(gs[-1, :-1])
#x_hist_ax = fig.add_subplot(gs[-2, -1])
#y_hist_ax = fig.add_subplot(gs[-1, -1])

#x_hist_ax.get_yaxis().set_visible(False)
#y_hist_ax.get_yaxis().set_visible(False)
#x_hist_ax.get_xaxis().set_visible(False)
#y_hist_ax.get_xaxis().set_visible(False)
residual_y_lim = (-6, 7)

x_res_ax.set_ylim(residual_y_lim)

y_res_ax.set_ylim(residual_y_lim)
# x_res_ax.set_yticks([-5, 0,5])
# y_res_ax.set_yticks([-5, 0,5])

#plot settings for seminar graph
main_plot_ax.spines['bottom'].set_color('white')
main_plot_ax.spines['top'].set_color('white') 
main_plot_ax.spines['right'].set_color('white')
main_plot_ax.spines['left'].set_color('white')

x_res_ax.spines['bottom'].set_color('white')
x_res_ax.spines['top'].set_color('white') 
x_res_ax.spines['right'].set_color('white')
x_res_ax.spines['left'].set_color('white')

y_res_ax.spines['bottom'].set_color('white')
y_res_ax.spines['top'].set_color('white') 
y_res_ax.spines['right'].set_color('white')
y_res_ax.spines['left'].set_color('white')

main_plot_ax.tick_params(axis='x', colors='white')
main_plot_ax.tick_params(axis='y', colors='white')
main_plot_ax.yaxis.label.set_color('white')
#main_plot_ax.xaxis.label.set_color('white')

x_res_ax.tick_params(axis='x', colors='white')
x_res_ax.tick_params(axis='y', colors='white')
x_res_ax.yaxis.label.set_color('white')
x_res_ax.xaxis.label.set_color('white')

y_res_ax.tick_params(axis='x', colors='white')
y_res_ax.tick_params(axis='y', colors='white')
y_res_ax.yaxis.label.set_color('white')
y_res_ax.xaxis.label.set_color('white')



#fitting gaussians
main_plot_ax.plot(amplitude_range[range_to_fit_start:range_to_fit_end]*1000,gaus(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, *popt_x),label='Gaussian fit for beam waist in x', color = "#FDE725")#3b528b")
main_plot_ax.plot(amplitude_range[range_to_fit_start:range_to_fit_end]*1000,gaus(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, *popt_y),label='Gaussian fit for beam waist in y', color = "#5ec962")
main_plot_ax.errorbar(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, wx_average[range_to_fit_start:range_to_fit_end], yerr=wx_err_plot[range_to_fit_start:range_to_fit_end], marker='o', linestyle='', label='Beam waist in x', color="#FDE725")#3b528b")
main_plot_ax.errorbar(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, wy_average[range_to_fit_start:range_to_fit_end], yerr=wy_err_plot[range_to_fit_start:range_to_fit_end], marker='o', linestyle='',  label= 'Beam waist in y', color='#5ec962')
main_plot_ax.set_ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
#main_plot_ax.set_xlabel('Amplitude / Milliwaves')
legend = main_plot_ax.legend()
plt.setp(legend.get_texts(), color='w')
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0,0,0,0))


#normalised residuals
x_norm_residuals=[]
y_norm_residuals=[]

x_norm_residuals = (wx_average - gaus(amplitude_range*1000,*popt_x))/wx_err_plot
y_norm_residuals = (wy_average - gaus(amplitude_range*1000,*popt_y))/wy_err_plot

x_res_ax.scatter(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, x_norm_residuals[range_to_fit_start:range_to_fit_end], color="#FDE725", marker='x')
#x_res_ax.stem(amplitude_range*1000, x_norm_residuals, linefmt=None, markerfmt='x',basefmt="k")
y_res_ax.scatter(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, y_norm_residuals[range_to_fit_start:range_to_fit_end], color='#5ec962', marker='x')
#y_res_ax.stem(amplitude_range*1000, y_norm_residuals, linefmt='C1', markerfmt='C1x', basefmt='k')
x_res_ax.axhline(y=0, c='w' )
y_res_ax.axhline(y=0, c='w')
x_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='#440154', alpha=0.15)
y_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='#440154', alpha=0.15)
#y_res_ax.set_xlabel('Milliwaves')
y_res_ax.set_xlabel('Amplitude / Milliwaves')



plt.text(-0.12, -0.1, "Norm. Residuals",color='white', verticalalignment="center", horizontalalignment="center", transform=x_res_ax.transAxes, rotation=90)

#plot histogram
#x_hist_ax.hist(x_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='blue')
#y_hist_ax.hist(y_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='orange')

#finding the minimum beam waist from the fit
# dots_x,dots_y,dots_y_err = amplitude_range, wx_average, yerr


print(min(wx_average))
z_x = np.where(wx_average== min(wx_average))
z_y = np.where(wy_average == min(wy_average))
print(z_y)
print(z_x)
print(amplitude_range[z_x]*1000)
print(amplitude_range[z_y]*1000)
#print(standard_error(wx_average))
#print(standard_error(wx_average))
#print(standard_error(wy_average))



plt.figure()
plt.scatter(range(0, len(amplitude_range)),max_pixel[0:len(amplitude_range)])
plt.xlabel('Image Number')
plt.ylabel('Value of Maximum Pixel in Image')




fig.savefig('seminar.png', transparent=True)



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

#plot of maximum intensity vs amplitude
# plt.figure(figsize=(10,4))
# plt.plot(amplitude_range,gaus(amplitude_range,*popt_x),label='Gaussian Fit')
# plt.plot(amplitude_range,gaus(amplitude_range,*popt_y),label='Gaussian Fit')
# plt.errorbar(amplitude_range, x_max_intensity_average, yerr=standard_error(x_max_intensity_average), marker='o', linestyle='', label='Peak Intensity in x', color='#3b528b')
# plt.errorbar(amplitude_range , y_max_intensity_average, yerr=standard_error(y_max_intensity_average), marker='o', linestyle='',  label= 'Peak Intensity in y', color='#5ec962')

# plt.ylabel('Intensity /')
# plt.xlabel('Amplitude /')
# plt.legend()

#plot of maximum intensity vs amplitude

# plt.figure(figsize=(9,8))
# plt.errorbar(amplitude_range*1000, x_max_intensity_average, yerr=standard_error(x_max_intensity_average), marker='o', linestyle='', label='Peak Intensity in x', color='blue')
# plt.errorbar(amplitude_range*1000, y_max_intensity_average, yerr=standard_error(y_max_intensity_average), marker='o', linestyle='',  label= 'Peak Intensity in y', color='orange')
# plt.ylabel('Intensity / Wm$^{-2}$')
# plt.xlabel('Milliwaves')

fig = plt.figure(figsize=(6,6))

#plot settings for seminar graph
ax = fig.add_subplot(111)

ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

#plt.plot(amplitude_range[range_to_fit_start:range_to_fit_end]*1000,gaus(amplitude_range[range_to_fit_start:range_to_fit_end]*1000,*popt_y),label='Gaussian Fit')
plt.errorbar(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, x_max_intensity_average[range_to_fit_start:range_to_fit_end], yerr=x_intensity_standard_err[range_to_fit_start:range_to_fit_end], marker='o', linestyle='', label='Peak Intensity in x', color='#FDE725')
plt.errorbar(amplitude_range[range_to_fit_start:range_to_fit_end]*1000, y_max_intensity_average[range_to_fit_start:range_to_fit_end], yerr=y_intensity_standard_err[range_to_fit_start:range_to_fit_end], marker='o', linestyle='',  label= 'Peak Intensity in y', color='#5ec962')
plt.ylabel('Integrated Intensity / Pixels', color = 'w')
plt.xlabel('Amplitude / Milliwaves', color='w')
legend = plt.legend()
plt.setp(legend.get_texts(), color='w')
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0,0,0,0))


# print(max(x_max_intensity_average))
# print(max(y_max_intensity_average))
# z_x = np.where(x_max_intensity_average== max(x_max_intensity_average))
# z_y = np.where(y_max_intensity_average ==max(y_max_intensity_average))
# print(z_x)
# print(z_y)
# print(amplitude_range[35])
# print(amplitude_range[36])


fig.savefig('seminar2.png', transparent=True)

plt.show()