####
#Code designed to use functions from the SLM class to run 'diagnostic' tests prior to running a set of measurements#
####

import numpy as np
from holograms.zernike import zernike
from holograms.gratings import hori_gradient, hori
from holograms.apertures import circ
from holograms.arrays import aags
from holograms.lenses import focal_plane_shift
# from beam_fitting_code import profile
# from beam_fitting_code import image
from camera import ImageHandler,Camera
import matplotlib.pyplot as plt
from slm import SLM
import time

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
exposure = 0.2
repeat = 0 

#path where images are saved for lens plot UPDATE!
path = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\December\18\Measure 10"


slm = SLM()
camera = Camera()
save_image = ImageHandler()

#test a grating with varied period to ensure the first order is being imaged
#this should move the position of the beam in the images. 
def grating_test(gradient):
    
    for gradient in range(gradient-30, gradient+30, 5):

        grating = hori_gradient(gradient)

        #apply hologram to SLM
        slm.apply_hologram(grating)

        #pause to allow for grating to load
        time.sleep(0.5)

        #use camera class to take and save photos
        camera.update_exposure(exposure)
        image = camera.take_image()
        save_image.save(image)

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
        time.sleep(1)

#test lens over a range of focal lengths to ensure the image is in focus
def in_focus(lens_shift):
    for shifts in range(lens_shift-10, lens_shift+10,1):
        lens = focal_plane_shift(
            shift=shifts,
            x0=X0,
            y0=Y0,
            wavelength=WAVELENGTH,
            pixel_size=SLM_PIXEL_SIZE,
            shape=SLM_SHAPE)

        #apply hologram to SLM
        slm.apply_hologram(lens)

        #pause to allow for grating to load
        time.sleep(0.5)

        #use camera class to take and save photos
        camera.update_exposure(exposure)
        image = camera.take_image()
        save_image.save(image)

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
        time.sleep(1)

        # #initialise profile class
        # prof=profile(path)

        # #create empty arrays for pixel values
        # x_intensities=[]
        # y_intensities=[]
        # x_max_intensities=[]
        # y_max_intensities=[]
        # max_pixel=[]

        # #crop it and get beam radius
        # z, wx, wy = prof.analyseBeamProfile(path)

        # plt.plot(range(lens_shift-10, lens_shift+10,1), wx)
        # plt.xlabel("Beam Radius in x")
        # plt.ylabel("Beam Radius in y")


#Takes a single image of three traps so ROI can be checked. 
def three_trap_ROI(X,Y):
    
    trap_hologram = aags(traps= ((X[0],Y[0]), (X[1],Y[1]),(X[2],Y[2])),
                        iterations=20, #must be greater than no. of traps
                        beam_waist=None,
                        beam_center=(256,256),
                        shape=(512,512)) 

    #apply hologram to SLM
    slm.apply_hologram(trap_hologram)

    #pause to allow for grating to load
    time.sleep(0.5)

        # #use camera class to take and save photos
    camera.update_exposure(exposure)
    image = camera.take_image()
    save_image.save(image)

        # Sleep for 1 second to allow for side-effects
        # such as the camera releasing its handles
        # to the OS
    time.sleep(1)

in_focus(-3)



