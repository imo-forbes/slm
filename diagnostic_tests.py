####
#Code designed to use functions from the SLM class to run 'diagnostic' tests prior to running a set of measurements#
####

import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, hori
from holograms.apertures import circ
from holograms.arrays import aags
from holograms.lenses import focal_plane_shift
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

slm = SLM()
#camera = Camera()
#save_image = ImageHandler()

#test a grating with varied period to ensure the first order is being imaged
#this should move the position of the beam in the images. 
def grating_test(gradient):
    
    for gradient in range(gradient-10, gradient+10, 1):

        grating = hori_gradient(gradient=HORI_GRADIENT)

        #apply hologram to SLM
        slm.apply_hologram(grating)

        #pause to allow for grating to load
        time.sleep(0.5)

        # #use camera class to take and save photos
        # camera.update_exposure(exposure)
        # image = camera.take_image()
        # save_image.save(image)

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
        time.sleep(1)