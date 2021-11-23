# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:07:50 2021

@author: imoge
"""

import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile

profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3")
end = 23
#def main():
for i in range (3,4,1):
    
    d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure {}". format(i)
    
    gradient_range = np.arange(0,300,10)
    
    z, wx, wy = profile.analyseBeamProfile(d)
    print(wx)
    
    w_x = []
    w_y = []
    for p in range(0, len(wx),1):

        w = wx[p] - wx[0]
        w_x.append(w)
        
    for q in range(0, len(wy), 1):
        w = wy[q] - wy[0]
        w_y.append(w)
        

    #plotting beam radius wx vs gradient range. First values removed to remove zero error
    plt.figure()
    plt.plot(gradient_range[1:end], wx[1:end], label='Beam waist in x')
    plt.plot(gradient_range[1:end], wy[1:end], label= 'Beam waist in y')
    plt.plot(gradient_range[1:end], w_y[1:end], label= 'Beam waist in y corrected')
    plt.plot(gradient_range[1:end], w_x[1:end], label= 'Beam waist in x corrected for background')
    plt.ylabel('Beam $\\frac{1}{e^2} $ waist in x / mm')
    plt.xlabel('Gradient on SLM')
    #plt.axvspan(200,300, 0, 2, alpha=0.25, color ='green') #bar for consideration at supervision
    plt.legend()
    plt.title('Beam $\\frac{1}{e^2} $ waists vs Gradient for Measurement '+ str(i))