"""A class to generate holograms for arrays of multiple Gaussian beams. This 
class manages the SLM and camera together to iteratively improve the hologram.
"""

import numpy as np
import holograms as hg
from scipy.fft import fft2,ifft2,fftshift,ifftshift
import cv2
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log10,floor,ceil
import pandas as pd

class ArrayGenerator():
    def __init__(self,slm,camera,extra_holos=None,shape=(512,512),
                 circ_aper_center=None):
        """Generates and iteratively improves Gerchberg Saxton holograms for 
        arrays of Gaussian traps.

        Parameters
        ----------
        slm : SLM
            Initialised SLM class to display holograms on.
        camera : Camera
            ThorLabs DCC1545M-GL camera class to capture images with.
        extra_holos : array, optional
            Extra holograms in the range 0-1 to apply to any generated 
            holograms. This will normally contain a blazed grating and perhaps 
            a Fresnel lens.
        shape : tuple of int
            The resolution of the SLM in pixels (x,y).
        circ_aper_center : tuple of float
            The center of a circular aperture applied in a blazed grating in 
            extra_holos. This is taken into account in the incident intensity 
            on the SLM.
        
        Returns
        -------
        None
        """
        self.slm = slm
        self.cam = camera
        self.shape = shape
        self.circ_aper_center = circ_aper_center
        self.extra_holos = extra_holos

        self.trap_df = pd.DataFrame()
        self.traps = None
        self.T = np.zeros(self.shape)
        self.phi = None
        self.psi_N = None
        
        self.generate_input_intensity()

    def generate_input_intensity(self,waist=None,center=None):
        """Defines the Gaussian intensity incident onto the SLM and stores it 
        as a parameter.

        Parameters
        ----------
        waist : float, optional
            The 1/e^2 waist of the beam, in SLM pixels. if waist is None, then 
            a uniform intensity is assumed.
        center : tuple of float, optional
            The center of the beam, in SLM pixels. if center is None, the beam 
            is centered on the SLM.

        Returns
        -------
        None
        """
        xx,yy = np.meshgrid(np.arange(self.shape[0]),np.arange(self.shape[1]))
        if waist is None:
            I = np.ones(self.shape)
        else:
            if center is None:
                center = tuple(x/2 for x in self.shape)
            xx = xx - center[0]
            yy = yy - center[1]
            r = np.sqrt(xx**2+yy**2)
            I = np.exp(-2*r**2/waist**2)
            I = I/np.max(I)
        if self.circ_aper_center is not None:
            I = hg.apertures.circ(I,self.circ_aper_center)
        self.input_intensity = I

    def set_traps(self,traps):
        """Set the location of the Gaussian traps to be used in the array.

        Parameters
        ----------
        traps : list of tuple of int
            A list containing the tuples (y,x) of the locations of the traps.
        
        Returns
        -------
        None
        """
        self.traps = traps
    
    def generate_initial_hologram(self,iterations=50):
        """Calculates the adaptive additive Gerchberg Saxton hologram for a 
        given list of uniform traps to create the initial Gaussian trap array 
        before any depth optimisation.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations that the AAGS algorithm should run for.
        
        Returns
        -------
        None
            The created hologram is saved to the object's array_holo parameter.
        """
        for trap in self.traps:
            self.T[trap] = 1
        N = len(self.traps)
        print('Initial array generation: {} traps'.format(N))
        phi = (np.random.rand(*self.shape))*2*np.pi
        prev_g = np.ones(self.shape)

        for i in range(iterations):
            print(i)
            u_plane = fftshift(fft2(np.sqrt(self.input_intensity)*np.exp(1j*phi)))
            B = np.abs(u_plane)
            psi = np.angle(u_plane)
            if i == N:
                print('i==N, saving psi for all future iterations')
                self.psi_N = psi
            elif i>N:
                print('i>N, using psi from i==N')
                psi = self.psi_N
            
            B_N = 0
            for trap in self.traps:
                B_N += B[trap]
            B_N /= N
            
            g = np.zeros(self.shape)
            for trap in self.traps:
                g[trap] = B_N/B[trap]*prev_g[trap]
            B = self.T*g
            prev_g = g
            x_plane = ifft2(ifftshift(B * np.exp(1j*psi)))
            phi = np.angle(x_plane)
        if self.psi_N is None:
            print('Warning: did not run for more iterations than traps.')
            print('Saving the last generated psi as psi_N.')
            self.psi_N = psi
        self.phi = phi
        print(self.phi)
        self.array_holo = (phi%(2*np.pi))/2/np.pi
        if self.circ_aper_center is not None:
            self.array_holo = hg.apertures.circ(self.array_holo,self.circ_aper_center)
        if self.extra_holos is not None:
            self.array_holo = (self.array_holo+self.extra_holos)%1

    def get_trap_depths(self,reps=1,plot=False):
        """Applies the pre-generated hologram onto the SLM and uses the 
        pre-obtained trap_df to fit Gaussians to each of the traps in the 
        array.

        Parameters
        ----------
        reps : int, optional
            The number of times the hologram should be applied before the 
            resultant trap depths are averaged and saved.
        plot : bool, optional
            Whether the camera image and the fitted array should be plotted 
            each time the array is fitted.

        Returns
        -------
        None
            The resultant trap depths are saved in the object's trap_df
            parameter.
        """

        I0ss = []
        for i in range(reps):
            self.slm.apply_hologram(self.array_holo)
            time.sleep(0.5)
            self.cam.auto_gain_exposure()
            image = self.cam.take_image()
            self.slm.apply_hologram(hg.blank())
            time.sleep(0.5)
            bgnd = self.cam.take_image()
            image.add_background(bgnd)
            array = image.get_bgnd_corrected_array()
            I0s = self.find_traps_df(array,plot)
            I0ss.append(I0s)
        I0ss = np.asarray(I0ss)
        I0s = np.average(I0ss, axis=0)
        I0s = list(I0s)
        self.trap_df['I0'] = I0s

    def generate_corrected_hologram(self,iterations=50):
        """Calculates the corrected adaptive additive Gerchberg Saxton hologram
        using the measured trap depths to aim for a more uniform trap array.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations that the AAGS algorithm should run for.
        
        Returns
        -------
        None
            The created hologram is saved to the object's array_holo parameter.
        """
        I0_N = self.trap_df['I0'].mean()
        for i, row in self.trap_df.iterrows():
            x = int(row['holo_x'])
            y = int(row['holo_y'])
            trap = (y,x)
            self.T[trap] *= np.sqrt(I0_N/row['I0'])
        N = len(self.traps)
        print('Corrected array generation: {} traps'.format(N))
        prev_g = np.ones(self.shape)
        phi = self.phi

        for i in range(iterations):
            print(i)
            u_plane = fftshift(fft2(np.sqrt(self.input_intensity)*np.exp(1j*phi)))
            B = np.abs(u_plane)
            print('Using psi from i==N')
            psi = self.psi_N
            
            B_N = 0
            for trap in self.traps:
                B_N += B[trap]/self.T[trap]
            B_N /= N
            
            g = np.zeros(self.shape)
            for trap in self.traps:
                g[trap] = (B_N/B[trap]*self.T[trap])*prev_g[trap]
            B = self.T*g
            prev_g = g
            x_plane = ifft2(ifftshift(B * np.exp(1j*psi)))
            phi = np.angle(x_plane)
        self.phi = phi
        self.array_holo = (phi%(2*np.pi))/2/np.pi
        if self.circ_aper_center is not None:
            self.array_holo = hg.apertures.circ(self.array_holo,self.circ_aper_center)
        if self.extra_holos is not None:
            self.array_holo = (self.array_holo+self.extra_holos)%1

    def gaussian2D(self,xy_tuple,amplitude,x0,y0,wx,wy,theta):
        (x,y) = xy_tuple
        sigma_x = wx/2
        sigma_y = wy/2 #function defined in terms of sigmax, sigmay
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()

    def find_traps_cv(self,array,plot=False):
        min_distance_between_traps = 30
        super_threshold_indices = array < 0
        array[super_threshold_indices] = 0
        img = np.uint8(array)
        blurred_img = cv2.GaussianBlur(img,(3,3),1)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        (thresh, bwimg) = cv2.threshold(blurred_img, 20, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(bwimg,cv2.HOUGH_GRADIENT,1,min_distance_between_traps,
                                    param1=200,param2=8,minRadius=10,maxRadius=20)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        if plot:
            cv2.imshow('detected circles',cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        popts = []
        for i,circle in enumerate(circles[0]):
            print(i)
            x0,y0,r = circle
            print(x0,y0)
            xmin = max(round(x0-min_distance_between_traps*0.75),0)
            xmax = min(round(x0+min_distance_between_traps*0.75),array.shape[1])
            ymin = max(round(y0-min_distance_between_traps*0.75),0)
            ymax = min(round(y0+min_distance_between_traps*0.75),array.shape[0])
            print(xmin,ymin,xmax,ymax)
            roi = array[ymin:ymax,xmin:xmax]
            max_val = np.max(array)
            x,y = np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
            popt, pcov = curve_fit(self.gaussian2D, (x,y), roi.ravel(), p0=[max_val,x0,y0,r,r,0])#,0])
            popts.append(popt)
            perr = np.sqrt(np.diag(pcov))
            data_fitted = self.gaussian2D((x,y),*popt).reshape(roi.shape[0],roi.shape[1])
            for arg,val,err in zip(self.gaussian2D.__code__.co_varnames[2:],popt,perr):
                prec = floor(log10(err))
                err = round(err/10**prec)*10**prec
                val = round(val/10**prec)*10**prec
                if prec > 0:
                    valerr = '{:.0f}({:.0f})'.format(val,err)
                else:
                    valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
                print(arg,'=',valerr)

            print('\n')

        if plot:
            fitted_img = np.zeros_like(array)
            x, y = np.meshgrid(np.arange(array.shape[1]), range(array.shape[0]))
            for popt in popts:
                fitted_img += self.gaussian2D((x,y),*popt).reshape(array.shape[0],array.shape[1])

            fig, (ax1,ax2) = plt.subplots(1, 2)
            fig.set_size_inches(9, 5)
            fig.set_dpi(100)
            c1 = ax1.pcolormesh(x,y,array,cmap=plt.cm.gray,shading='auto')
            ax1.invert_yaxis()
            ax1.set_title('camera image')
            fig.colorbar(c1,ax=ax1,label='pixel count')
            c2 = ax2.pcolormesh(x,y,fitted_img,cmap=plt.cm.viridis,shading='auto')
            ax2.invert_yaxis()
            ax2.set_title('fitted array')
            fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
            fig.tight_layout()
            plt.show()
        
        return popts

    def find_traps_df(self,array,plot=False,width=15,min_distance_between_traps=30):
        popts = []
        I0s = []
        for i, row in self.trap_df.iterrows():
            print(i)
            x0 = row['img_x']
            y0 = row['img_y']
            r = width
            print(x0,y0)
            xmin = max(round(x0-min_distance_between_traps*0.75),0)
            xmax = min(round(x0+min_distance_between_traps*0.75),array.shape[1])
            ymin = max(round(y0-min_distance_between_traps*0.75),0)
            ymax = min(round(y0+min_distance_between_traps*0.75),array.shape[0])
            print(xmin,ymin,xmax,ymax)
            roi = array[ymin:ymax,xmin:xmax]
            max_val = np.max(array)
            x,y = np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
            popt, pcov = curve_fit(self.gaussian2D, (x,y), roi.ravel(), p0=[max_val,x0,y0,r,r,0])#,0])
            popts.append(popt)
            I0s.append(popt[0])
            perr = np.sqrt(np.diag(pcov))
            data_fitted = self.gaussian2D((x,y),*popt).reshape(roi.shape[0],roi.shape[1])
            for arg,val,err in zip(self.gaussian2D.__code__.co_varnames[2:],popt,perr):
                prec = floor(log10(err))
                err = round(err/10**prec)*10**prec
                val = round(val/10**prec)*10**prec
                if prec > 0:
                    valerr = '{:.0f}({:.0f})'.format(val,err)
                else:
                    valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
                print(arg,'=',valerr)
            try:
                self.trap_df.loc[i,'I0'] += popt[0]
            except:
                self.trap_df.loc[i,'I0'] = popt[0]
            print('\n')

        if plot:
            fitted_img = np.zeros_like(array)
            x, y = np.meshgrid(np.arange(array.shape[1]), range(array.shape[0]))
            for popt in popts:
                fitted_img += self.gaussian2D((x,y),*popt).reshape(array.shape[0],array.shape[1])

            fig, (ax1,ax2) = plt.subplots(1, 2)
            fig.set_size_inches(9, 5)
            fig.set_dpi(100)
            c1 = ax1.pcolormesh(x,y,array,cmap=plt.cm.gray,shading='auto')
            ax1.invert_yaxis()
            ax1.set_title('camera image')
            fig.colorbar(c1,ax=ax1,label='pixel count')
            c2 = ax2.pcolormesh(x,y,fitted_img,cmap=plt.cm.viridis,shading='auto')
            ax2.invert_yaxis()
            ax2.set_title('fitted array')
            fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
            fig.tight_layout()
            plt.show()
        return I0s

    def get_single_trap_coords(self):
        for i,trap in enumerate(self.traps):
            self.trap_df.loc[i,'holo_x'] = trap[1]
            self.trap_df.loc[i,'holo_y'] = trap[0]
            print(i)
            self.array_holo = self._generate_aags_hologram([trap],5)     
        
            if self.extra_holos is not None:
                holo = (self.array_holo+self.extra_holos)%1
            else:
                holo = self.array_holo
            
            self.slm.apply_hologram(holo)
            time.sleep(0.1)
            self.cam.auto_gain_exposure()
            
            image = self.cam.take_image()
            time.sleep(0.1)
            self.slm.apply_hologram(hg.blank())
            bgnd = self.cam.take_image()
            image.add_background(bgnd)
            
            array = image.get_bgnd_corrected_array()
            popts = self.find_traps(array)
            if len(popts) > 1:
                print('Warning: multiple circles found for trap {}'.format(trap))
            popt = popts[0]
            self.trap_df.loc[i,'img_x'] = popt[1]
            self.trap_df.loc[i,'img_y'] = popt[2]
    
    def save_trap_df(self,filename):
        self.trap_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        self.trap_df['holo_y'] = self.trap_df['holo_y'].astype(int)
        self.trap_df.to_csv(filename,float_format='{:f}'.format,encoding='utf-8')

    def load_trap_df(self,filename):
        self.trap_df = pd.read_csv(filename,index_col=0)
        self.trap_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        self.trap_df['holo_y'] = self.trap_df['holo_y'].astype(int)
