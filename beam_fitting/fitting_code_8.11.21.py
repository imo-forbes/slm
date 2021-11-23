# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:37 2021

@author: imoge
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import sys
sys.path.append('Z:\Tweezer\People\Imogen\GitHub\slm\beam_fitting')
from function_fits import Gaussian as G
from PIL import Image
from scipy.optimize import curve_fit

def arraySorter(*args):
    c = np.zeros([len(args[0]),len(args)])
    for i in range(len(args)):
        c[:,i] = args[i]
    c = c[c[:,0].argsort()]

    lnargs = [None] * int(len(args))  
    for i in range(len(args)):
        lnargs[i] =  c[:,i]
        
    return lnargs

''' gaussian fit'''
class Gaussian:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.a, self.x0, self.w0, self.d = self.fitVals()[0]

        self.ea, self.ex0, self.ew0, self.ed = self.fitVals()[1]

    def f(self, x,a,x0,w0, d):
        return d + a*np.exp(-2*(x-x0)**2/w0**2)

    def fitVals(self):
        peak = np.max(self.y)
        fwhm = peak
        i = 0
        
        while (fwhm-np.min(self.y)) > (peak-np.min(self.y)) / 2:
            fwhm = self.y[(np.argmax(self.y)+i)]
            i += 1
            if (np.argmax(self.y)+i == (len(self.y)-1)):
                break

        e2_width = 2 * (self.x[(np.argmax(self.y)+i)]-self.x[(np.argmax(self.y))])
        
        p0 = [(np.max(self.y)-np.min(self.y)),self.x[np.argmax(self.y)], e2_width, np.min(self.y)]

        popt,pcov = curve_fit(self.f,self.x,self.y,p0=p0, maxfev=int(1e5))    
   
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def applyFit(self, xth):
        popt, pcov = self.fitVals()
        return self.f(xth, self.a, self.x0, self.w0, self.d)
    
class image():
    def __init__(self, imvals = None, pixelSize=1):
        self.imvals = imvals
        self.integratedFit = True
        
        self.xpix = None
        self.ypix = None
        self.xInt = None
        self.yInt = None
        self.xSlice = None
        self.ySlice = None
        self.pixelSize = pixelSize
    
    def openImage(self, fname, directory=r'Z:\Tweezer\People\Imogen\GitHub\slm\images\2021\November\08\Measure_6', fileType = 'png'):
        """Can load the image from a file"""
        if directory != None:
            os.chdir(directory)
        if fileType == 'png':
           self.imvals = np.array(Image.open(fname))#[:,:,0] #comment out this slice for other file formats
        elif fileType == 'ascii':
             self.imvals = np.loadtxt(fname)[:,1:]
       # print(np.shape(self.imvals))
    
    def integrateImage(self):
        """Sum along both axes of the 2D image to get integrated counts"""
        xInt = np.zeros(len(self.imvals[0,:]))
        yInt = np.zeros(len(self.imvals[:,0]))
        for i in range(len(xInt)):
            xInt[i] = np.sum(self.imvals[:,i])
        for j in range(len(yInt)):
            yInt[j] = np.sum(self.imvals[j,:])
        
        self.xInt = xInt
        self.yInt = yInt
        
    def cropImage(self,crop_x,crop_y):
        """Crop the image"""
        self.maxPixel()
        self.imvals = self.imvals[(self.yc-crop_y):(self.yc+crop_y),(self.xc-crop_x):(self.xc+crop_x)]
        
    def sliceImage(self):
        """Take a slice through the maximum pixel in x and y"""
        self.maxPixel()
        self.xSlice = self.imvals[int(self.yc), :]
        self.ySlice = self.imvals[:,int(self.xc)]

    def getPixels(self):
        """Get array of pixels for the x and y directions"""
        xLen = np.zeros(len(self.imvals[0,:]))
        yLen = np.zeros(len(self.imvals[:,0]))
        self.xpix = np.arange(0, len(xLen), 1) * self.pixelSize
        self.ypix = np.arange(0, len(yLen), 1) * self.pixelSize

    def maxPixel(self):
        """Find the maximum pixel in the image"""
        self.yc, self.xc = np.unravel_index(self.imvals.argmax(), self.imvals.shape)     
    def rotate_image(self,angle):
        self.imvals = ndimage.rotate(self.imvals,angle)

    
    def fitImage(self):
        """Fit the image with a Gaussian"""
        self.getPixels()
        self.sliceImage()
        
        self.integrateImage()

        self.gx = Gaussian(self.xpix, self.xInt)
        print('wx =  ', self.gx.w0)
        print('ewx = ', self.gx.ew0)
        self.gy = Gaussian(self.ypix, self.yInt)   
        print('wy =  ', self.gy.w0)
        print('ewy = ', self.gy.ew0)   



class profile():
    def __init__(self, directory, pixelSize=5.2e-3):
        self.directory = directory
        self.pixelSize = pixelSize
        print("Pixel Size =",self.pixelSize*1e3,"um")
        
    def analyseBeamProfile(self, fit = True, plot_all=False,crop_x=None,crop_y=None,angle=None):
        """Loop through the image directory and fit every image."""
        
        z = np.array([])
        wx = np.array([])
        wy = np.array([])
        
        
        i = 0
        print(os.listdir(self.directory))
        for filename in os.listdir(self.directory):
            
            if filename[-4:] == '.png':
               # print(filename[:-4])
                print("Pixel Size =",self.pixelSize*1e3,"um")
                im = image(pixelSize=self.pixelSize)
                im.openImage(filename)
                if crop_x!=None:
                    im.cropImage(crop_x,crop_y)
                if angle!=None:
                    im.rotate_image(angle)
                im.fitImage()
                #print(im.gx.w0)
                if plot_all==True:
                    self.plotSingle(filename,crop_x=crop_x,crop_y=crop_y,angle=angle)
                z = np.append(z, float(filename[:-4]))
                wx = np.append(wx, im.gx.w0)
                wy = np.append(wy, im.gy.w0)
            
          #  print(str(i),end=' ')
            i +=1
        #print("")
        #print(z)
        
        z = z#-9.5 - 2.
        
        z, wx, wy = arraySorter(z, wx, wy)
        
        plt.figure(figsize=(10,4))
        plt.plot(z, wx, marker = 'o', linestyle = 'none', color = '#AA2B4A', label='Plots z, wx')
        plt.plot(z, wy, marker = 'o', linestyle = 'none', label = 'Plots z, wy')
       # plt.xlim(6.5,9.)
        
        if fit == 'hyperbola':
            self.fitHyperbola(z, wx,'x', clr =  '#AA2B4A')
            self.fitHyperbola(z, wy,'y', clr = '#006388')
        elif fit=='linear':
            self.fitLinear(z, wx,'x', clr =  '#AA2B4A')
            self.fitLinear(z, wy,'y', clr = '#006388')
            
        
        plt.xlabel('camera position / cm')
        plt.ylabel('beam 1/e2 waist / mm')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        return (z, wx, wy)

    def plotZ(self, fit = True):
        """Loop through the image directory and fit every image."""
        
        z = np.array([])
        wx = np.array([])
        wy = np.array([])
        
        i = 0
        for filename in os.listdir(self.directory):
            if filename[-4:] == '.bmp':
                im = image(pixelSize=self.pixelSize)
                im.openImage(filename)
                im.fitImage()
                z = np.append(z, float(filename[:-4]))
                wx = np.append(wx, im.gx.w0)
                wy = np.append(wy, im.gy.w0)
                
            i +=1

        
        z, wx, wy = arraySorter(z, wx, wy)
        
        plt.figure(figsize=(10,4))
        plt.plot(z, wx, marker = 'o', linestyle = 'none', color = '#AA2B4A')
        plt.plot(z, wy, marker = 'o', linestyle = 'none')
        print(z)
        print(wy)
        

profile = profile(r'Z:\Tweezer\People\Imogen\GitHub\slm\images\2021\November\08\Measure 7')
profile.analyseBeamProfile()


