import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy.fft import fft2,ifft2,fftshift,ifftshift
import numpy as np
from holograms.arrays import aags
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

exposure = 100

def input_intensity(waist,center):
        xx,yy = np.meshgrid(np.arange(14),np.arange(10))
        xx = xx - center[0]
        yy = yy - center[1]
        
        r = np.sqrt(xx**2+yy**2)
        I = np.exp(-2*r**2/waist**2)
        I = I/np.max(I)
        return I

def gerchbert_saxton(input_intensity,desired_image,iterations):
    shape = desired_image.shape
    slm_amp = np.sqrt(input_intensity)
    slm_phase = np.random.rand(*shape)
    img_amp = np.sqrt(desired_image)
    img_phase = np.zeros(shape)
    
    for i in range(iterations):
        print(i)
        slm = slm_amp * np.exp(1j*slm_phase)
        img = fftshift(fft2(slm))
        img_phase = np.angle(img)
        img = img_amp * np.exp(1j*img_phase)
        slm = ifft2(ifftshift(img))
        slm_phase = np.angle(slm)
    
    return slm_phase

# def weighted_gerchbert_saxton(input_intensity,desired_image,iterations):
#     shape = desired_image.shape
#     slm_amp = np.sqrt(input_intensity)
#  #Ar
#     slm_phase = np.random.rand(*shape)
#     img_amp = np.sqrt(desired_image)#target - At
#     img_phase = np.zeros(shape)
#     weight = np.zeros((10,10))
#     for i in range(0,10,1):
#         for j in range(0,10,1):   
#             weight[i,j] = math.exp((np.subtract(img_amp[i,j],slm_amp[i,j])))
    
#     for i in range(iterations):
#         print(i)
#         print(weight)
#         slm = weight * np.exp(1j*slm_phase)
#         img = fftshift(fft2(slm))
#         img_phase = np.angle(img)
#         img = np.exp(1j*img_phase)
#         slm = ifft2(ifftshift(img))
#         slm_phase = np.angle(slm)
    
#     return slm_phase

img = Image.open(r"Z:\Tweezer\People\Imogen\GitHub\slm\arrays\alien.png").convert('L')
array = np.asarray(img)
print(np.shape(img))
I = input_intensity(336, (288,277))
#holo = weighted_gerchbert_saxton(I,array,100)
holo = gerchbert_saxton(I,array,100)

# plt.pcolor(img)
# plt.colorbar()
# #plt.show()

# plt.figure()
# #plt.pcolor(holo)
# #plt.colorbar()

# plt.figure()
# plt.pcolor(holo_2)
# plt.colorbar()
# #plt.show()

cam_img = np.abs(fft2(np.sqrt(I)*np.exp(1j*holo)))**2
cam_img /= np.max(cam_img)
plt.pcolor(cam_img)
plt.colorbar()
#plt.show()

#code for applying to SLM
# camera = Camera()
# save_image = ImageHandler()
# camera.update_exposure(exposure)
slm = SLM()

#apply hologram to SLM
slm.apply_hologram(holo)

#pause to allow for grating to load
time.sleep(1)

#use camera class to take and save photos
#camera.update_exposure(exposure)
# image = camera.take_image()
# save_image.save(image)

# # Delete causes camera disconnect
# del camera
# del save_image

# Sleep for 1 second to allow for side-effects
# such as the camera releasing its handles
# to the OS
time.sleep(1)


