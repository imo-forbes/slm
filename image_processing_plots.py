# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:35:37 2021

@author: imoge
"""

import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile
from beam_fitting_code import image
import scipy.optimize as scpo

#file images are being taken from
d = "./images/2021/November/22/Measure 16"
amplitude_range= np.arange(0, 0.3, 0.005)  

#intialise modules
profile=profile(d)

#collect beam radius arrays from images
z, wx, wy = profile.analyseBeamProfile(d)

#collect intensity values from images
for x in range(0,(3*len(amplitude_range)),1):
    
    d = d + "\{}.png". format(x)
    
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle()

#remove background - code makes image 0 the background image
wx_corrected = []
wy_corrected = []
for image_no in range(1,len(wx),1):
    wx_corrected.append(wx[image_no] - wx[0])
    wy_corrected.append(wy[image_no] - wy[0])
    

    
#breaks list of values into repeats and averages
amplitude_range= np.arange(0, 0.3, 0.005) 
images_in_set = len(amplitude_range)
wx_1 = wx[1:(images_in_set+1)]
wx_2 = wx[(images_in_set+1):2*(images_in_set)+1]
wx_3 = wx[2*(images_in_set)+1::]
wx_average = (wx_1 + wx_2 + wx_3)/3

wy_1 = wy[1:(images_in_set+1)]
wy_2 = wy[(images_in_set+1):2*(images_in_set)+1]
wy_3 = wy[2*(images_in_set)+1::]
wy_average = (wy_1 + wy_2 + wy_3)/3

#error analysis
#TO DO


#plot of beam waist vs amplitude
plt.figure(figsize=(10,4)) 
plt.scatter(amplitude_range, wx_average, label='Mean beam waist in x')
#plt.scatter(np.arange(-1, 1,0.1), wx_corrected[1:21], label='Beam waist in x')
plt.scatter(amplitude_range , wy_average, label= 'Mean beam waist in y')
plt.ylabel('Beam $\\frac{1}{e^2} $ waist in x / mm')
plt.xlabel('Amplitude ')
#plt.axvspan(200,300, 0, 2, alpha=0.25, color ='green') #bar for consideration at supervision
plt.legend()
plt.show()

#plot of beam waist vs milliwaves
plt.figure(figsize=(10,4))
plt.scatter(amplitude_range*1000, wx_average, label='Mean beam waist in x')
plt.scatter(amplitude_range*1000, wy_average, label='Mean beam waist in y')
plt.ylabel('Beam $\\frac{1}{e^2} $ waist in x / mm')
plt.xlabel('Milliwaves ')
plt.legend()
plt.show()
