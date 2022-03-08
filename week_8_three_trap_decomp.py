import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from holograms.arrays import aags
from holograms.zernike import zernike
from holograms.gratings import grating
from mpl_toolkits.mplot3d import Axes3D

#BEGIN CONFIGURATION 
#r used to set highest radial order to be considered.
#note that np.range used to for order 4 to be considered, r = 4+1 = 5 for example
r = 17
R = [5,9,13,17,21]
#define Zernike polynomials to fit to:

ZERNIKE_COORDINATES = []

#generating zernike polynomials
for radial in range(r):
    for azimuthal in np.arange(-radial,radial+2,2):
        ZERNIKE_COORDINATES.append((radial, azimuthal))

bar_text_color = 'k'
X = [236,256,276, 256]
Y = [256,256,256, 236]

three_traps = aags(((X[0],Y[0]), (X[1], Y[1]), (X[2], Y[2])),iterations=30,beam_waist=None,beam_center=(288,227),shape=(512,512))

#END CONFIGURATION
plt.figure()
plt.pcolor(three_traps)

plt.figure()
plt.plot(three_traps)

from scipy import fftpack
fourier = np.fft.fftshift(np.fft.fft2(three_traps))
plt.figure()
plt.imshow((abs(fourier)))
#print(fourier)


plt.figure()
plt.plot(abs(fourier))

inv_fourier = np.fft.fftshift(np.fft.ifft2(fourier))
plt.figure()
plt.imshow((abs(inv_fourier)))
loc = np.where(fourier > 1)
abs_ft = abs(fourier[loc])

#print(inv_fourier[loc])
dist = np.linspace(0,512,1)
wave = 0
for x in abs_ft:
    wave += x/2*np.sin(2*np.pi*dist)
    #print(wave)

plt.figure()
plt.plot(grating(100,0))
 # for other plots I changed to


plt.show()