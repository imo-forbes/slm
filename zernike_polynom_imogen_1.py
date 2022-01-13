# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:32:51 2021

@author: imoge
"""

# import functions from directory
import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.mixing import mix
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time
import warnings

r = 0 #sets radial component of Zernike Polynom
a = 0 #sets azimuthal component of Zernike Polynom

amplitude_range = np.arange(-1, 1, 0.05) 
exposure = 3
    
#initialise classes            
#camera = Camera()
#save_image = ImageHandler()
slm = SLM()

#take background image before SLM set and save for analysis
# camera.update_exposure(exposure)
# image = camera.take_image()
# save_image.save(image)


#take multiple runs for each setting
for i in range(0,3,1):

    # looping function- calls gradient function and then uses camera to take photo
    for x in amplitude_range:
    
        #Apply polynomials/lenses/gratings
        hor_grating_1 = hori_gradient(gradient=-7.3)
        hor_grating_2 = hori(period = 7)
        #vert_grating = vert_gradient(gradient=5.29, shape=(512,512))
        zernike_polynom = zernike(radial=r,azimuthal=a,amplitude=x,x0=233,y0=251,radius=None,shape=(512,512))
        lens = focal_plane_shift(shift=-3,x0=233,y0=251,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512))
    
        #Sum holograms together
        x = lens + zernike_polynom + hor_grating_1 + hor_grating_2

        holo = circ(x, x0=233, y0=251, radius = 233)

        #mix holograms
        h = mix(holo)
    
        #apply hologram to SLM
        slm.apply_hologram(h)
    
    
        #pause to allow for grating to load
        time.sleep(0.5)
    
    
        #use camera class to take photo
        camera.update_exposure(exposure)
        image = camera.take_image()
    
        #array = image.get_array()
        #take_image calls Image which adds properties to save when the photo is saved
    
        #save photo
    
        #save_image.show_image(array) #can edit out to speed up image taking
        save_image.save(image)

        #check if it's saturated
        # if np.amax(image) == 255:
        #     warnings.warn("Maximum pixel is saturated")
        # elif np.amax(image) >= 225:
        #     warnings.warn("Maximum pixel value close to saturation")
        # elif np.amax(image) <=50:
        #     warnings.warn("Maximum pixel value is below 50")
        
    i = i+1
