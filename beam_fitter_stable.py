import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import sys
sys.path.append('Z:\Tweezer\People\Alex\Programs')
from function_fits import Gaussian as G
from PIL import Image
from scipy.optimize import curve_fit

'''.
10/9/2019
Script to fit beam profiles
    
'''

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


'''hyperbolic fit'''
class Hyperbola:
    def __init__(self, x, y, wavelength = 1):
        self.x, self.y = x, y
        #self.wavelength = wavelength /1000
        self.z0, self.w0, self.zr, self.y0 = self.fitVals()
    
    def f(self, z, z0, w0, zr, y0):
        return w0*np.sqrt(1+((z-z0)/zr)**2)#+y0

    def fitVals(self):
        initial_guess = [self.x[np.argmin(self.y)], np.min(self.y), 1 , 0]
       # print(initial_guess)

        popt,pcov = curve_fit(self.f,self.x,self.y,initial_guess, maxfev=80000)
        return popt

    def applyFit(self, zth):
        return self.f(zth, self.z0, self.w0, self.zr, self.y0)

''' linear fit'''
class Linear:
    def __init__(self, x, y, guess=None):
        self.x, self.y = x, y
        self.m, self.c = self.fitVals()[0]
        self.em, self.ec = self.fitVals()[1]

    def f(self, x, m, c):
        return m * x + c

    def fitVals(self):
        p0 = [1,1]
        popt,pcov = curve_fit(self.f,self.x,self.y,p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def applyFit(self, xth):
        return self.f(xth, self.m, self.c)


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
    
    def openImage(self, fname, directory=None, fileType = 'bmp'):
        """Can load the image from a file"""
        if directory != None:
            os.chdir(directory)
        if fileType == 'bmp':
            self.imvals = np.array(Image.open(fname))[:,:,0] #comment out this slice for other file formats
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
        # print('wx =  ', self.gx.w0)
        # print('ewx = ', self.gx.ew0)
        self.gy = Gaussian(self.ypix, self.yInt)   
        # print('wy =  ', self.gy.w0)
        # print('ewy = ', self.gy.ew0)   

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
            
            if filename[-4:] == '.bmp':
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
        plt.plot(z, wx, marker = 'o', linestyle = 'none', color = '#AA2B4A')
        plt.plot(z, wy, marker = 'o', linestyle = 'none')
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


    def fitHyperbola(self, x, y, xy, clr):
        H = Hyperbola(x, y)
       # print(H.fitVals())
        
        xfit = np.linspace(min(x), max(x), 1000)
        yfit = H.applyFit(xfit)

        if xy == 'x':
            label = 'z0$_x$ = '+str(np.around(H.z0, 3))+', w0$_x$ = '+str(np.around(H.w0, 3))+', z$_{rx} = $'+str(np.around(H.zr, 3))
        elif xy == 'y':
            label = 'z0$_y$ = '+str(np.around(H.z0, 3))+', w0$_y$ = '+str(np.around(H.w0, 3))+', z$_{ry} = $'+str(np.around(H.zr, 3)) 
        plt.plot(xfit, yfit, label=label, color = clr)
        return(xfit, yfit)
    
    
    def fitLinear(self, x, y, xy, clr):
        L = Linear(x, y)

        xfit = np.linspace(min(x), max(x), 1000)
        yfit = L.applyFit(xfit)
       # print(L.m, L.em)
      #  print(L.c, L.ec)
        if xy == 'x':
            label = 'm$_x$ = '+str(np.around(L.m, 3))+', c$_x$ = '+str(np.around(L.c, 3))
        elif xy == 'y':
            label = 'm$_y$ = '+str(np.around(L.m, 3))+', c$_y$ = '+str(np.around(L.c, 3))
        
        plt.plot(xfit, yfit, label=label, color = clr)
        return(xfit, yfit)
        

    def plotSingle(self, filename, single = True,crop_x=None,crop_y=None,angle=None):
        """Plot on realisation of the MOT"""
        if single == True:
            plt.figure()

        im = image(pixelSize=self.pixelSize)
        im.openImage(filename)
        if crop_x!=None:
            im.cropImage(crop_x,crop_y)
        if angle!=None:
            im.rotate_image(angle)
        im.fitImage()
    

        xth = np.linspace(np.min(im.xpix), np.max(im.xpix), 4*len(im.xpix))
        yth = np.linspace(np.min(im.ypix), np.max(im.ypix), 4*len(im.ypix))
        xopt = im.gx.applyFit(xth)
        yopt = im.gy.applyFit(yth)
        
        ax1 = plt.subplot2grid((6,6), (1,1), rowspan=5, colspan = 5)
        ax2 = plt.subplot2grid((6,6), (1,0), rowspan=5, colspan = 1)
        ax3 = plt.subplot2grid((6,6), (0, 1), rowspan=1, colspan = 5)
        plt.subplots_adjust(left = 0.27, right = 0.85)
        # top gaussian
        ax3.plot(im.xpix, im.xInt)
    
        ax3.xaxis.tick_top()
        ax3.set_xlabel('x / mm')
        ax3.xaxis.set_label_position('top')
        ax3.set_ylabel('Intensity')
        ax3.set_yticks((0, np.around(np.around(np.max(im.xInt),-1)/2), np.around(np.max(im.xInt),-1)))
        ax3.set_xlim([np.min(im.xpix), im.xpix[-1]])
        ax3.set_ylim([np.min(im.xInt), np.max(im.xInt)])
        ax3.plot(xth, xopt)
        
        # side gaussian
        ax2.invert_xaxis()
        ax2.plot(im.yInt,im.ypix)
        ax2.plot(yopt, yth)
    
        ax2.set_ylabel('y / mm')
        ax2.set_xlabel('Intensity')
        ax2.set_xticks((0, np.around(np.max(im.ypix),-1)))
        ax2.set_ylim([0, im.ypix[-1]])
        ax2.set_xlim([np.max(im.yInt), np.min(im.yInt)])
        ax2.invert_yaxis()
  
        ax1.imshow(im.imvals, interpolation='nearest', extent = [0, im.xpix[-1], 0,  im.ypix[-1]])
        ax1.yaxis.set_label_position('right')
        
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('y / mm')
        
        ax1.yaxis.tick_right()
        
        ax1.legend(loc = 'upper left')
        #print(self.MOTx0, self.xc)
        #print(self.MOTy0, self.yc)
        
        if single == True:
            plt.show()


if __name__ == '__main__':

    imagePath = r"Z:\Tweezer\Experimental Results\2021\June\25"
    os.chdir(imagePath)
    plt.close('all')
    p = profile(imagePath, pixelSize=5.2e-3)#5.2e-3 # 5.86
    
   # Z,waist_x,waist_y=p.analyseBeamProfile('hyperbola',plot_all=True,angle=0)
    p.plotSingle('after_1st_lens.bmp',crop_x=400,crop_y=400,angle=0)











    

