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
import pandas as pd

#CONFIGURATION
X0 = 233
Y0 = 251
SLM_SHAPE = (512, 512)
SLM_PIXEL_SIZE = 15e-6
WAVELENGTH = 1064e-9
APERTURE_RADIUS = 233
HORI_GRADIENT = -7.3
HORI_PERIOD = 7
LENS_SHIFT = -3

#set variables for run
amplitude_range = np.arange(-1,1,0.05)
exposure = 5
repeat = 0 

#set polynomials to loop through. r is radial coord, a is azimuthal.
# exclude 1, 0 and defocus for final hologram 
radial_coords = [4,4,4,4,4,3,3,3,3,2,2]
azimuthal_coords = [4,2,0,-2,-4,3,1,-1,-3,2,-2]
amplitude_corrections_x = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

#END CONFIGURATION

def main(): 

    #initialise classes            
    # camera = Camera()
    # save_image = ImageHandler()
    slm = SLM()

    #take background image before SLM set and save for analysis
    # camera.update_exposure(exposure)
    # image = camera.take_image()
    # save_image.save(image)

    #create final hologram:
    zernike_holo=0
    for value in range(1,len(radial_coords)):
        x = zernike(
            radial=radial_coords[value],
            azimuthal=azimuthal_coords[value],
            amplitude=amplitude_corrections_x[value],
            x0=X0,
            y0=Y0,
            radius=None,
            shape=SLM_SHAPE)
        print(radial_coords[value], azimuthal_coords[value], amplitude_corrections_x[value])
        zernike_holo = zernike_holo + x

    #other gratings and lens to apply
    hor_grating_1 = hori_gradient(gradient = HORI_GRADIENT)
    hor_grating_2 = hori(period = HORI_PERIOD)
    lens = focal_plane_shift(
        shift=LENS_SHIFT,
        x0=X0,
        y0=Y0,
        wavelength=WAVELENGTH,
        pixel_size= SLM_PIXEL_SIZE,
        shape = SLM_SHAPE)

    #Sum holograms together
    final_holo_sum = lens + hor_grating_1 + hor_grating_2 #+ zernike_holo
    #apply aperture
    final_holo_aperture = circ(final_holo_sum, x0= X0, y0= Y0, radius = APERTURE_RADIUS)

    #take multiple runs for each setting
    for repeat in range(0,3,1):
        
        #apply hologram to SLM
        slm.apply_hologram(final_holo_aperture)
        
        
        #pause to allow for grating to load
        time.sleep(0.5)
        
        
        #use camera class to take photo
        # camera.update_exposure(exposure)
        # image = camera.take_image()
        
        # #array = image.get_array()
        # #take_image calls Image which adds properties to save when the photo is saved
        
        # #save photo
        # #save_image.show_image(array) #can edit out to speed up image taking
        # save_image.save(image)
        
        repeat = repeat + 1

if __name__ == "__main__":
    main()