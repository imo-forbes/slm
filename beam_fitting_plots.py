import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile
from beam_fitting_code import image
import scipy.optimize as scpo


image = image(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3")
profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3")
end = 30
#def main():
#for i in [22,27,28]:
    
d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\08\Measure 3" 
    
    #gradient_range = np.arange(0,300,10)
    
    
z, wx, wy = profile.analyseBeamProfile(d)
   # min_val_x = wx.min()
   # min_val_y = wy.min()
   # print(min_val_x)
   # print(min_val_y)
   # z_x = np.where(wx==min_val_x)
    #z_y = np.where(wy == min_val_y)
   # print(z_x)
   # print(z_y)
   # print(z[20])


    
 

    #plotting beam radius wx vs gradient range. First values removed to remove zero error
plt.figure()
plt.scatter(range(0,200,10), wx[0:20], label='Beam waist in x')
plt.scatter(range(0,200,10), wy[0:20], label= 'Beam waist in y')
plt.ylabel('Beam $\\frac{1}{e^2} $ waist in x / mm')
plt.xlabel('Gradient')
#plt.axvspan(200,300, 0, 2, alpha=0.25, color ='green') #bar for consideration at supervision
plt.legend()
    #plt.title('Beam $\\frac{1}{e^2} $ waists vs Gradient for Measurement '+ str(i))
    
#best fit line
def Line(gradient, intercept,x): 
    return gradient*x + intercept
# Performing linear regression on values 40
actual_fit_parameters, error= scpo.curve_fit(Line,range(0,200,10), wx[0:20])
fit_intercept = actual_fit_parameters[0]
fit_gradient = actual_fit_parameters[1] 
ybestfit = Line(fit_gradient,fit_intercept, wx[0:20])
plt.plot(ybestfit)

    #plotting camera position vs gradient
    #plt.figure()
    #plt.plot(z, gradient_range)
    #plt.ylabel('Gradient on SLM')
    #plt.xlabel('Camera Position / cm ')
    #plt.title('Gradient on SLM vs Camera Position for Measurement '+ str(i))
    
    #looping through images to get intensities
x_max_intensities=[]
y_max_intensities=[]
x_intensities = []
y_intensities = []
x_pixels=[]
y_pixels =[]
x_th=[]
y_th=[]
x_opt=[]
y_opt=[]
for x in range(0,2,1):
    
    d = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 30\{}.png". format(x)
    
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle()
    #
    x_intensities.append(xInt)
    y_intensities.append(yInt)
    x_max_intensities.append(max(xInt))
    y_max_intensities.append(max(yInt))

print(x_max_intensities)
print(y_max_intensities)
    #
     #   x_max_index = np.argmax(xInt)
      #  y_max_index = np.argmax(yInt)
    #
       # x_pixels.append(xpix[x_max_index])
     #   y_pixels.append(ypix[y_max_index])
    #
     #   x_opt.append(xopt)
      #  y_opt.append(yopt)
        #x_th.append(xth)
        #y_th.append(yth)
amplitude_range = np.arange(-10,10,0.1)    
plt.scatter( amplitude_range, x_max_intensities, label='Intensity in x', c='blue')
#plt.plot(range(-100,100,5), x_max_intensities, c ='blue')
#plt.scatter(range(-100,100,5), y_max_intensities, label='Intensity in y', c='orange')
plt.scatter(amplitude_range, y_max_intensities, c='orange', label = 'Intensity in y')
plt.xlabel('Amplitude')
plt.ylabel('Peak Intensity')
plt.legend()
print(max(x_max_intensities))
print(max(y_max_intensities))
z_x = np.where(x_max_intensities== max(x_max_intensities))
z_y = np.where(y_max_intensities == max(y_max_intensities))
print(z_x)
print(z_y)
print(amplitude_range[154])
print(amplitude_range[48])
#print(z[21])
#print(z[19])
        #plt.title('Intensity for Measurement '+ str(i) + ' Image '+ str(x) + '.png') 
        #plt.figure()
        #plt.plot(ypix, yInt)
        #plt.xlabel('y /mm')
        #plt.ylabel('Intensity in y')
        #plt.title('Intensity for Measurement '+ str(i) + ' Image '+ str(x) + '.png')

#plots using intensity
#if __name__ == "__main__":
    #main()

    