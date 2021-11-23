# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:27:12 2021

@author: imoge
"""

import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from beam_fitting_code import profile

profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 22")
end = 23
#def main():
   
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 22"

    
amplitude_range = np.arange(-100,100,5)
    
z_1, wx_1, wy_1 = profile.analyseBeamProfile(d)
    
profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 27")  
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 27"
    

    
z_2, wx_2, wy_2 = profile.analyseBeamProfile(d)

profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 28")
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 28"
    

    
z_3, wx_3, wy_3 = profile.analyseBeamProfile(d)


z= (z_1+z_2+z_3)/3
wx =(wx_1 + wx_2 + wx_3)/3
wy = (wy_1+ wy_2+wy_3)/3

#av_wx = stat.mean(wx[1:end])
#av_wy = stat.mean(wy[1:end])

#plotting beam radius wx vs gradient range. First values removed to remove zero error
plt.figure()
plt.plot(amplitude_range, wx, label='Beam waist in x')
plt.plot(amplitude_range, wy, label= 'Beam waist in y')
plt.ylabel('Beam $\\frac{1}{e^2} $ waist in x / mm')
plt.xlabel('Gradient on SLM')
plt.axhline(av_wx, color='red', linestyle='--', label='Average Beam Waist in x')
plt.axhline(av_wy, color='black', linestyle=':', label='Average Beam Waist in y')
    #plt.axvspan(200,300, 0, 2, alpha=0.25, color ='green') #bar for consideration at supervision
plt.legend()
plt.title('Beam $\\frac{1}{e^2} $ waists vs Gradient for Average of Measurements 3 and 4')