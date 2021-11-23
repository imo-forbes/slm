# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:01 2021

@author: imoge
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from beam_fitting_code import profile
from beam_fitting_code import image
import scipy.optimize as scpo
from matplotlib import gridspec


image = image(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 4")
profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 4")
end = 30
#def main():
#for i in [22,27,28]:
    
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 4" 
    
    #gradient_range = np.arange(0,300,10)
    
    
z, wx, wy = profile.analyseBeamProfile(d)
print(wx)
print(wy)

wx_3 =[0.0675426, 0.06707095, 0.06605362, 0.06950791, 0.07155753, 0.07325261,
 0.07423863, 0.074663 ,  0.07228283, 0.07043498, 0.06903209, 0.06877585,
 0.06685861, 0.06476439, 0.06266323, 0.06045583, 0.05579585, 0.0541417,
 0.04832877, 0.04367291, 0.03957176, 0.03598452, 0.03044468, 0.02731693,
 0.02404667,0.02231639, 0.02115917, 0.02051597, 0.01974897, 0.01921663]
wy_3 = [0.05892095, 0.07506375, 0.0768042,  0.0788341,  0.07762701, 0.08050195,
 0.08123295, 0.08453032, 0.08451976, 0.08536302, 0.0851472,  0.09081183,
 0.08936404, 0.09112116, 0.09143186, 0.10191775, 0.09798257, 0.1025126,
 0.1000355,  0.10573241, 0.09697089, 0.09617724, 0.20639645, 0.21422507,
 0.22440351, 0.08239186, 0.22119566, 0.06242776, 0.21089749, 0.21255814]
wx_4 = [0.06928623, 0.06751807, 0.06594411, 0.0702314,  0.07169117, 0.07358709,
 0.0741002,  0.07467814, 0.07206717, 0.07035735, 0.0684402,  0.06931861,
 0.06739503, 0.06519417, 0.06302999, 0.06038294, 0.05650993, 0.05437833,
 0.04822673, 0.04380165, 0.0398731,  0.03603173, 0.03041496, 0.02738068,
 0.02425136, 0.02242383, 0.02129294, 0.02054266, 0.019783,   0.01912316]
wy_4 = [0.05660813, 0.07545516, 0.07570472, 0.07895648, 0.07824831, 0.0807372,
 0.08312752, 0.08465688, 0.08251967, 0.08590555, 0.08612581, 0.09063968,
 0.09139771, 0.09186137, 0.09331078, 0.09508592 ,0.0967387,  0.1063706,
 0.09945499, 0.10838829, 0.09501668 ,0.18258994, 0.21104898, 0.21524978,
 0.21776395, 0.06510763, 0.06630059, 0.22939584, 0.21443569, 0.21446705]


#errors
def stand_error(x):
    y = np.std(x)
    z= math.sqrt(20)
    return y/z

#best fit line
def Line(gradient, intercept,x): 
    return gradient*x + intercept

# Performing linear regression on values
run1_actual_fit_parameters, errors = scpo.curve_fit(Line,range(0,200,10), wx_3[0:20])
run1_fit_intercept = run1_actual_fit_parameters[0]
run1_fit_gradient = run1_actual_fit_parameters[1] 
run1_ybestfit = Line(run1_fit_gradient, run1_fit_intercept, range(0,200,10))

run2_actual_fit_parameters, errors = scpo.curve_fit(Line,range(0,200,10), wy_3[0:20])
run2_fit_intercept = run2_actual_fit_parameters[0]
run2_fit_gradient = run2_actual_fit_parameters[1] 
run2_ybestfit = Line(run2_fit_gradient, run2_fit_intercept, range(0,200,10))



#grid to include residuals

fig = plt.figure(figsize=(9,12))
gs = gridspec.GridSpec(7, 6, hspace=0,wspace=0)

main_plot_ax = fig.add_subplot(gs[:-2, :-1])
run1_res_ax = fig.add_subplot(gs[-2, :-1])
run2_res_ax = fig.add_subplot(gs[-1, :-1])
run1_hist_ax = fig.add_subplot(gs[-2, -1])
run2_hist_ax = fig.add_subplot(gs[-1, -1])

run1_hist_ax.get_yaxis().set_visible(False)
run2_hist_ax.get_yaxis().set_visible(False)
run1_hist_ax.get_xaxis().set_visible(False)
run2_hist_ax.get_xaxis().set_visible(False)
residual_y_lim = (-4, 3)

run1_res_ax.set_ylim(residual_y_lim)
run2_res_ax.set_ylim(residual_y_lim)
run1_res_ax.set_yticks([-2, 0,2])
run2_res_ax.set_yticks([-2, 0,2])
#main_plot_ax.set_yticklabels(([-100,0,100,200,300,400,500,600,700]),fontsize= 16)
#run1_res_ax.set_yticklabels(([-2, 0, 2]),fontsize= 16)
#run2_res_ax.set_yticklabels(([-2,0,2]),fontsize= 16)
#run2_res_ax.set_xticklabels([-3.2,-3,-2.8,-2.6,-2.4,-2.2,-2],fontsize= 16)
    

    #plotting beam radius wx vs gradient range. First values removed to remove zero error

main_plot_ax.errorbar(range(0,200,10), (wx_3[0:20]) , yerr= stand_error(wx_3), marker= 'o', linestyle='',  label='Beam waist in x')
main_plot_ax.errorbar(range(0,200,10), (wy_3[0:20]), yerr= stand_error(wy_3), marker ='o', linestyle ='' , label= 'Beam waist in y')
main_plot_ax.set_ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
main_plot_ax.set_xlabel('Gradient')
#main_plot_ax.axhline(y=0.06505469449999998, color='darkblue', linestyle='--', label='Average Beam Waist in x')
#main_plot_ax.axhline(y=0.0869727465, color='darkgoldenrod', linestyle='--', label='Average Beam Waist in y')

main_plot_ax.plot(range(0,200,10), run1_ybestfit, color='darkblue', linestyle='--', label='Linear Fit in x')
main_plot_ax.plot(range(0,200,10), run2_ybestfit, color='darkgoldenrod', linestyle='--', label='Linear Fit in y')
main_plot_ax.legend(loc=2, prop={'size': 8})

main_plot_ax.get_shared_x_axes().join(main_plot_ax, run1_res_ax)
main_plot_ax.get_shared_x_axes().join(main_plot_ax, run2_res_ax)
main_plot_ax.set_xticklabels([])

run2_res_ax.set_xlabel("Gradient", fontsize=12)

# plotting norm residuals
run1_norm_residuals=[]
run2_norm_residuals=[]


run1_norm_residuals = (wx_3[0:20] - run1_ybestfit)/stand_error(wx_3)
run2_norm_residuals = (wy_3[0:20] - run2_ybestfit)/stand_error(wy_3)





plt.xlabel("ln(h)")
run1_res_ax.scatter(range(0,200,10), run1_norm_residuals, color='blue')
run1_res_ax.stem(range(0,200,10), run1_norm_residuals, linefmt=None, basefmt="k")
run2_res_ax.scatter(range(0,200,10), run2_norm_residuals, color='orange', marker='x')
run2_res_ax.stem(range(0,200,10), run2_norm_residuals, linefmt='C1', markerfmt='C1x', basefmt='k')
run1_res_ax.axhline(y=0, c='k' )
run2_res_ax.axhline(y=0, c='k')
run1_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='green', alpha=0.25)
run2_res_ax.axhspan(ymin=1, ymax=-1, xmin=0, xmax=1, color='green', alpha=0.25)



plt.text(-0.08, 0, "Normalised Residuals", verticalalignment="center", horizontalalignment="center", transform=run1_res_ax.transAxes, rotation=90, fontsize=12)

#plot histogram
run1_hist_ax.hist(run1_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='blue')
run2_hist_ax.hist(run2_norm_residuals, np.linspace(-1,1,8), orientation = 'horizontal', color='orange')
