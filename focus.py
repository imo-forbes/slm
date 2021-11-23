# import functions from directory
import numpy as np
from holograms.zernike import zernike
from holograms.gratings import grating_gradient, vert_gradient
from holograms.lenses import focal_plane_shift
from holograms.mixing import mix
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

camera = Camera()
save_image = ImageHandler()
slm = SLM()

lens_range= range(-15,-13,1)

for s in lens_range:
    lens = focal_plane_shift(shift=s,x0=233,y0=251,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512))
    print(s)

    slm.apply_hologram(lens)
    
    
    #pause to allow for grating
    time.sleep(0.5)
    
    
    #use camera class to take photo
    camera.update_exposure(3.5)
    image = camera.take_image()
    
    #array = image.get_array()
    #take_image calls Image which adds properties to save when the photo is saved
    
    #save photo
    
    #save_image.show_image(array) #can edit out to speed up image taking
    save_image.save(image)