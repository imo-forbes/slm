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

#initialise classes            
# camera = Camera()
# save_image = ImageHandler()
slm = SLM()

#take background image before SLM set and save for analysis
# camera.update_exposure(exposure)
# image = camera.take_image()
# save_image.save(image)

#create final hologram:
EXCEL_SHEET_NAME = "Correction_Alt1" #"Correction Hologram FINAL"
df = pd.read_excel (r'C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\data_for_hologram_7.12.xlsx', sheet_name=[EXCEL_SHEET_NAME])[EXCEL_SHEET_NAME]
r = df['r'].tolist()
a = df['a'].tolist()
amplitude = df['amplitude'].tolist()

zernike_holo=0
for i in range(1,len(r)):
    x = zernike(radial=r[i],azimuthal=a[i],amplitude=amplitude[i],x0=233,y0=251,radius=None,shape=(512,512))
    print(r[i], a[i], amplitude[i])
    zernike_holo = zernike_holo + x

#other gratings and lens to apply
hor_grating_1 = hori_gradient(gradient=-7.3)
hor_grating_2 = hori(period = 7)
#zernike_polynom = zernike(radial=r,azimuthal=a,amplitude=x,x0=233,y0=251,radius=None,shape=(512,512))
lens = focal_plane_shift(shift=-3,x0=233,y0=251,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512))

#Sum holograms together
x = lens + zernike_holo + hor_grating_1 + hor_grating_2

holo = circ(x, x0=233, y0=251, radius = 233)

#mix holograms
h = mix(holo)

#take multiple runs for each setting
for i in range(0,3,1):
    
    #apply hologram to SLM
    slm.apply_hologram(h)
    
    
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
    
    i = i+1
