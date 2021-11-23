# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:38:07 2021

@author: imoge
"""

# Importing Image class from PIL module
from PIL import Image
from beam_fitting_code import profile
import cv2
import os



def image_crop(x):
    
    # Opens a image in RGB mode
    im = Image.open(x)
 
    # Size of the image in pixels (size of original image)

    width, height = im.size


    # Setting the points for cropped image
    left = 2 * width/4
    top = height/4
    right = 3 * width/4
    bottom = 2* height/4
 
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
 
    # Shows the image in image viewer
    i=3
    filename='crop_measure_'+str(i)+'.png'
    im1.save(filename)
    cv2.imread(filename)
    path = r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\Cropped_Images_wk6" 
    (cv2.imwrite(os.path.join(filename), im1))
    
profile=profile(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 30")

#for i in range(0,200,1):
image_crop(r"C:\Users\imoge\OneDrive\Documents\Fourth Year\Project\Imogen\GitHub\slm\images\2021\November\15\Measure 30\3.png")