# -*- coding: utf-8 -*-

# import functions from directory
import numpy as np
from holograms.gratings import grating_gradient, vert_gradient
from holograms.lenses import focal_plane_shift
from holograms.mixing import mix
from camera import ImageHandler,Camera
from slm import SLM
import time

wavelength = 1064*10**-9
focal_length = 100 * 10 **-3
pixel_size = 15*10**-6
L = 7.68*10**-3

distance = np.arange(15, 7680, 255.5) #step size to allow 30 measurements to be taken

period_size = []
gradient_range=[]

#function to work out period_size for the various distanes
for x in distance:
    d = (wavelength*focal_length/(x*10**-6))
    period_size.append(d)
 

#function to work out the gradients for the various period sizes
for d in period_size:
    g=np.tan(np.arcsin(d/L))
    gradient_range.append(g) 

gradient_range = np.arange(0, 300, 10)
    
#initialise classes            
#camera = Camera()
#save_image = ImageHandler()
slm = SLM()


# looping function- calls gradient function and then uses camera to take photo
for g in gradient_range:
    
    #phase for SLM on gradient and lens
    hor_grating = grating_gradient(gradient=g,angle=0,shape=(512,512))
    vert_grating = vert_gradient(gradient=g, shape=(512,512))
    
    
    lens = focal_plane_shift(shift=-20,x0=233,y0=251,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512))
    
    #mix holograms
    h = mix(hor_grating)
    
    # add in apply hologram
    slm.apply_hologram(h)
    
    
    
    #pause to allow for grating
    time.sleep(0.5)
    
    #use camera class to take photo
    #camera.update_exposure(18)
    #image = camera.take_image()
    
    #array = image.get_array()
    #take_image calls Image which adds properties to save when the photo is saved
    
    #save photo
    
    #save_image.show_image(array) #can edit out to speed up image taking
    save_image.save(image)
    
    
    
    
    
    