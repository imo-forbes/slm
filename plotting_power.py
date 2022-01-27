import numpy as np
import matplotlib.pyplot as plt
from beam_fitting_code import profile
from beam_fitting_code import image


#wavelength set to value used in set-up
wavelength = 1064*10**-9

#r and z set to constant values as beam radii should not move with Zernike polynomials applied
r = 0.001
z = 0.001

#define rayleigh range, w(x) and the x values for the plot using formula from Grimm review
def rayleigh_range(wx):
    return (np.pi*wx**2)/wavelength

def w(wx):
    return wx * np.sqrt(1+ (z/rayleigh_range(wx))**2)

def x_value(wx):
    return -2/(np.pi*wx**2) * np.exp(-2 * r**2/(wx**2))


# read in images, set amplitude range used
d = "./images/2021/November/29/Measure 30"
amplitude_range= np.arange(-0.5, 0.5, 0.01) 
images_in_set = len(amplitude_range)


profile=profile(d)

#collect beam radius arrays from images
z, wx, wy = profile.analyseBeamProfile(d)

#break set into repeated runs and average
wx_1 = wx[1:(images_in_set+1)] 
wx_2 = wx[(images_in_set+1):2*(images_in_set)+1]
wx_3 = wx[2*(images_in_set)+1::]
wx_average = (wx_1 + wx_2 + wx_3)/3

#collect intensity values from images
x_intensities=[]
y_intensities=[]
x_max_intensities=[]
y_max_intensities=[]
max_pixel=[]


for x in range(0,(3*len(amplitude_range)+1),1):
    d =  r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\29\Measure 30" + "/{}.png". format(x)
    yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = profile.plotSingle(d)

        #take out max intensity values
    x_intensities.append(xInt)
    y_intensities.append(yInt)
    x_max_intensities.append(max(xInt))
    y_max_intensities.append(max(yInt))

#break into 3 repeats and average
x_max_intensity_1 = np.array(x_max_intensities[1:(images_in_set+1)])
x_max_intensity_2 = np.array(x_max_intensities[(images_in_set+1):2*(images_in_set)+1])
x_max_intenisty_3 = np.array(x_max_intensities[2*(images_in_set)+1::])
x_max_intensity_average = (x_max_intensity_1 + x_max_intensity_2 + x_max_intenisty_3)/3

#loop x values through formula 
x_vals = []
for i in wx_average:
    x = x_value(i)
    x_vals.append(x)


#plot intensity vs x(wx)
x_plot= x_vals
#print(len(x_plot), len(x_max_intensity_average))
plt.scatter(x_plot[1::], x_max_intensity_average[1::])

plt.ylabel("Intensity / Wmm$^{-2}$")
plt.xlabel(' A / mm$^2$')

# plt.figure()
# plt.scatter(wx_average[1::]**2, x_max_intensity_average[1::] )
plt.show()
    


