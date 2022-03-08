import numpy as np
from sympy import O
from holograms.zernike import zernike, R
from holograms.gratings import hori_gradient, vert_gradient, hori
from holograms.apertures import circ
from holograms.lenses import focal_plane_shift
from holograms.arrays import aags
from matplotlib import gridspec
from holograms.mixing import mix
import matplotlib.pyplot as plt
from slm import SLM
from scipy.optimize import minimize, curve_fit
import matplotlib
import time
import pandas as pd
from PIL import Image

#BEGIN CONFIGURATION 
#r used to set highest radial order to be considered.
#note that np.range used to for order 4 to be considered, r = 4+1 = 5 for example
r = 5
R = [5]
#define Zernike polynomials to fit to:

ZERNIKE_COORDINATES = []

#generating zernike polynomials
for radial in range(r):
    for azimuthal in np.arange(-radial,radial+2,2):
        ZERNIKE_COORDINATES.append((radial, azimuthal))



bar_text_color = 'k'
X = [236,256,276, 256]
Y = [256,256,256, 236]

three_traps = aags(((X[0],Y[0]), (X[1], Y[1]), (X[2], Y[2])),iterations=30,beam_waist=None,beam_center=(288,227),shape=(512,512))

#END CONFIGURATION

def main(): 
    for r in R:

        ZERNIKE_COORDINATES = []

        #generating zernike polynomials
        for radial in range(r):
            for azimuthal in np.arange(-radial,radial+2,2):
                ZERNIKE_COORDINATES.append((radial, azimuthal))

        #same as Zernike polynomial function in SLM code but removes the amplitude factor and the circular aperture
        def zernike_no_aperture_no_amp(radial=0,azimuthal=0,x0=None,y0=None,radius=None,shape=(512,512)):

            if azimuthal > radial:
                raise ValueError("azimuthal must be less than or equal to radial")
            if x0 is None:
                x0 = shape[0]/2
            if y0 is None:
                y0 = shape[1]/2
            if radius is None:
                radius = min([shape[0]-x0,shape[1]-y0,x0,y0])
            x = np.arange(shape[0])
            y = np.arange(shape[1])
            xx,yy = np.meshgrid(x,y)
            r = np.sqrt((xx-x0)**2+(yy-y0)**2)
            r /= radius
            phi = np.arctan2(yy-y0,xx-x0)
            if azimuthal >= 0:
                phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
            else:
                phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)

            return phase




        #same as Zernike polynomial function in SLM code but removes the circular aperture
        def zernike_no_aperture(amplitude = 1, radial=0,azimuthal=0,  x0=None, y0=None, radius=None, shape=(512,512)):

            if azimuthal > radial:
                raise ValueError("azimuthal must be less than or equal to radial")
            if x0 is None:
                x0 = shape[0]/2
            if y0 is None:
                y0 = shape[1]/2
            if radius is None:
                radius = min([shape[0]-x0,shape[1]-y0,x0,y0])
            x = np.arange(shape[0])
            y = np.arange(shape[1])
            xx,yy = np.meshgrid(x,y)
            r = np.sqrt((xx-x0)**2+(yy-y0)**2)
            r /= radius
            phi = np.arctan2(yy-y0,xx-x0)
            if azimuthal >= 0:
                phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
            else:
                phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)
            phase *= amplitude

            return phase
        coefficients = []
        def inital_guess(r):

            for radial in range(r):
                for azimuthal in np.arange(-radial,radial+2,2):
                    results_row = pd.DataFrame()
                    results_row.loc[0,'zernike_radial'] = radial
                    results_row.loc[0,'zernike_azimuthal'] = azimuthal
                    zernike_poly = zernike(radial,azimuthal,1)#wrap_phase=False)
                    contrib = trap_hologram*zernike_poly
                    coeff = np.sum(trap_hologram*zernike_poly)/np.sum(zernike_poly**2)
                    coefficients.append(coeff)
            
            return coefficients

        #creates linear sum of Zernike Polynomials to fit over
        def sum_zernike(coeffs):
            total_sum = 0

            for coeff, zernike in zip(coeffs, CACHED_ZERNIKE):
                total_sum += coeff * zernike

            return total_sum
            
        #finds the coefficients (amplitudes) of the Zernike Polynomials required to minimise 
        #a function of the sum of the Trap Hologram - Zernikes Sum 
        def zernike_decomposition(trap_hologram):

            def objective_func(x):
                return np.sum((trap_hologram - sum_zernike(x))**2) 


            amplitudes = minimize(objective_func, x0=np.zeros(len(ZERNIKE_COORDINATES)), bounds=[(-1, 1) for i in range(len(ZERNIKE_COORDINATES))])
            
            #print(amplitudes)
            #returns array of amplitdues for best fit
            return amplitudes.x,objective_func(amplitudes.x)


        CACHED_ZERNIKE = [zernike(rad, azi) for rad,azi in ZERNIKE_COORDINATES]

        X = [236,256,276, 256]
        Y = [256,256,256, 236]

        print("Starting trap hologram")

        trap_hologram = three_traps

        print("Generated trap hologram")

        #plot colourmap of input hologram
        plt.figure()
        plt.title('Colormap of Initial Hologram')
        plt.pcolor(trap_hologram)

        tic = time.perf_counter()
        decomp,_ = zernike_decomposition(trap_hologram)#(trap_hologram)
        toc = time.perf_counter()
        print(f"Completed fitting in {toc - tic:0.4f} seconds")
        print(decomp)
        print(_)

        # plot colormap of decomposition 
        final_periodic = sum_zernike(decomp)

        plt.figure()
        plt.title('Colormap of Hologram from Decomposition')
        plt.pcolor(final_periodic)

        # import cv2

        # for r in range(128,4, -1):
        #     for x in range(0 ,r//2):
        #         for y in range(0,r//2):
        #             slice_v = trap_hologram[y:y+r, x:x+r]

        #             resized_slice = cv2.resize(trap_hologram, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

        #             amps, score = zernike_decomposition(resized_slice)
        #             print("tried ",r,x,y,"with score",score)
        #             if score < 1:
        #                 break



        #additional settings for formatting for dark background
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        #use amplitudes to make a bar plot of contributions
        fig, ax = plt.subplots()   

        ax.spines['bottom'].set_color(bar_text_color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color(bar_text_color)
        ax.xaxis.label.set_color(bar_text_color)
        ax.tick_params(axis='x', colors=bar_text_color)


        bar_label =ZERNIKE_COORDINATES
        x = range(0,len(decomp),1)
        y=[]

        #convert to milliwaves
        for amp in decomp:
            y.append(amp*1000)

        for i in range(len(x)):
            ax.bar(x[i], y[i], color=(0.47, 0.76, 0.98), edgecolor = bar_text_color,  linewidth=1.5, label = bar_label[i])

        ax.tick_params(axis='y', colors=bar_text_color)
        plt.ylabel('Amplitude / Milliwaves', color = bar_text_color)
        plt.xlabel('Polynomial', color = bar_text_color)
        plt.title('Zernike Amplitude Components from Decomposition', color = bar_text_color)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(ZERNIKE_COORDINATES, rotation = 45) 


        plt.savefig('images_for_seminar/hologramcontributes.png', dpi=400, bbox_inches='tight', transparent = True)


        plt.show()


if __name__ == "__main__":
    main()
