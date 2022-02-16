import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from beam_fitting_code import profile
from beam_fitting_code import image
import scipy.optimize as scpo
from scipy.optimize import curve_fit
from matplotlib import gridspec
from pandas import plotting
import imageio
import cv2
import os
import csv
from rich.progress import track


from rich import print


#CONFIGURATION
X_start = [1,1,1,41,41,41,83,83,83]
Y_start = [1,42,84,1,41,82,1,36,79]
X_end = [21,21,21,61,61,61,104,104,104]
Y_end = [21,61,101,31,61,101,18,58,97]
amplitude_range = np.arange(-1,1,0.1)
wx_stand_dev = [0.0007, 0.0008, 0.0009]
wy_stand_dev =[0.0002, 0.0001, 0.0001]
int_x_stand_dev = [33,52,37]
int_y_stand_dev = [18,20,15]
start_range = 5
end_range = 15




print(amplitude_range)
images_in_set = len(amplitude_range)

#create empty arrays for pixel values
amplitude_corrections_x =[]
amplitude_corrections_y = []

#measurement numbers to analyse
measurements = np.arange(,7,1)

#END CONFIGURATION

def main(): 
    
    for measure_value in measurements:

    #set path to where images are saved
        path = r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\January\30\Measure {}".format(measure_value)

        # #create directories 
        # for file in range(0,9,1):
        #     try:
        #         os.mkdir(path + "/cropped_images_beam" + str(file+1))
        #     except OSError:
        #         print("[yellow bold blink]Warning: Directory already exists, so not creating ")
        
        # for image_no in track(range(0,31,1)):
        #     image = cv2.imread(r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\February\05\Measure {}\{}.png".format(measure_value, image_no),cv2.IMREAD_GRAYSCALE)

        #     for i in range(0,9,1):
        #         y=Y_start[i]
        #         x=X_start[i]
        #         h=Y_end[i]
        #         w=X_end[i]
        #         crop = image[y:h, x:w]
        #         cv2.imwrite(path + "/cropped_images_beam" + str(i+1) + "/" +str(image_no) + ".png", crop )

        # # #initialise profile class

        for beam_no in range(0,3,1):

            path = r"Z:\Tweezer\People\Imogen\GitHub\slm\images\2022\January\30\Measure {}\cropped_images_beam{}".format(measure_value, beam_no+1)
            prof=profile(path)

            x_intensities=[]
            y_intensities=[]
            x_max_intensities=[]
            y_max_intensities=[]
            max_pixel=[]

            #crop it and get beam radius
            try:
                z, wx, wy = prof.analyseBeamProfile(path) 
            except FileNotFoundError:
                print("[red bold]Cannot find file {}, skipping...".format(path))
                continue
            
            for x in range(0,(3*len(amplitude_range)+1),1):
                
                try:
                
                    d =  path + "/{}.png". format(x)
                except FileNotFoundError:
                    print("[red bold]Cannot find file {}, skipping...".format(path))
                    continue

               
                
                yInt, xInt, xpix, ypix, xth, xopt, yth, yopt = prof.plotSingle(d) #, crop_x=X_CROP[beam_no], crop_y=Y_CROP[beam_no])

            #find the maximum pixel
                pix = imageio.imread(d)
                max_pixel.append(np.amax(pix))
                if np.amax(pix) == 255:
                    print("[yellow blink bold]Warning: Maximum pixel is saturated in Image " + str(x))
                elif np.amax(pix) >= 245:
                    print("[yellow blink bold]Warning: Maximum pixel value close to saturation: Image "+ str(x) + " has max pixel value " + str(np.amax(pix)))
                elif np.amax(pix) <=50:
                    print("[yellow blink bold]Warning: Maximum pixel value is below 50 in Image " + str(x))

                #take out max intensity values
                x_intensities.append(xInt)
                y_intensities.append(yInt)
                x_max_intensities.append(max(xInt))
                y_max_intensities.append(max(yInt))

            #split image values into runs and average 
            wx_1 = wx[1:(images_in_set+1)] 
            wx_2 = wx[(images_in_set+1):2*(images_in_set)+1]
            wx_3 = wx[2*(images_in_set)+1::]
            wx_average = (wx_1+ wx_2+ wx_3)/3
            wx_std = np.std(np.vstack([wx_1, wx_2, wx_3]), axis=0)
            wx_standard_err = wx_std / np.sqrt(3)
            wx_err_quad = np.sqrt((wx_standard_err)**2 + (wx_stand_dev[beam_no])**2)

            wy_1 = wy[1:(images_in_set+1)]
            wy_2 = wy[(images_in_set+1):2*(images_in_set)+1]
            wy_3 = wy[2*(images_in_set)+1::]
            wy_average = (wy_1 + wy_2 + wy_3)/3
            wy_std = np.std(np.vstack([wy_1, wy_2, wy_3]), axis=0)
            wy_standard_err = wy_std/ np.sqrt(3)
            wy_err_quad = np.sqrt((wy_standard_err)**2 + (wx_stand_dev[beam_no])**2)
            
            #repeats and averaging intensities
            x_max_intensity_1 = np.array(x_max_intensities[1:(images_in_set+1)])
            x_max_intensity_2 = np.array(x_max_intensities[(images_in_set+1):2*(images_in_set)+1])
            x_max_intensity_3 = np.array(x_max_intensities[2*(images_in_set)+1::])
            x_max_intensity_average = (x_max_intensity_1 + x_max_intensity_2 + x_max_intensity_3)/3
            x_max_std = np.std(np.vstack([x_max_intensity_1, x_max_intensity_2, x_max_intensity_3]), axis =0 )
            x_max_std_err = x_max_std/np.sqrt(3)
            int_x_err_quad = np.sqrt((x_max_std_err)**2 + (int_x_stand_dev[beam_no])**2)

            y_max_intensity_1 = np.array(y_max_intensities[1:(images_in_set+1)])
            y_max_intensity_2 = np.array(y_max_intensities[(images_in_set+1):2*(images_in_set)+1])
            y_max_intensity_3 = np.array(y_max_intensities[2*(images_in_set)+1::])
            y_max_intensity_average = (y_max_intensity_1 + y_max_intensity_2 + y_max_intensity_3)/3
            y_max_std = np.std(np.vstack([y_max_intensity_1, y_max_intensity_2, y_max_intensity_3]), axis =0)
            y_max_std_err = y_max_std/np.sqrt(3)
            int_y_err_quad = np.sqrt((y_max_std_err)**2 + (int_x_stand_dev[beam_no])**2)
            
           

            #take minimum waist values
            print("Minimum beam waist in x = " + str(min(wx_average[start_range:end_range])))
            print("Minimum beam waist in y = " + str(min(wy_average[start_range:end_range])))
            z_x = np.argmin(wx_average[start_range:end_range])
            z_y = np.argmin(wy_average[start_range:end_range])
            amplitudes = amplitude_range[start_range:end_range]
            print("Amplitude correction for beam waist in x = " + str(amplitudes[z_x]))
            print("Amplitude correction for beam waist in y = " + str(amplitudes[z_y]))

            #create lists of the amplitude correction values
            amplitude_corrections_x.append(amplitudes[z_x])
            amplitude_corrections_y.append(amplitudes[z_y])
            print(amplitude_corrections_x)
            print(amplitude_corrections_y)

            #plot error bars of the minimum waist vs milliwave amplitude
            plt.figure()
            plt.errorbar(amplitudes*1000, wx_average[start_range:end_range], yerr=wx_err_quad[start_range:end_range], marker='o', linestyle='', label='Mean beam waist in x for beam' +str(beam_no+1), color='blue')
            plt.errorbar(amplitudes*1000, wy_average[start_range:end_range], yerr=wy_err_quad[start_range:end_range], marker='o', linestyle='',  label= 'Mean beam waist in y', color='orange')
            print(wx_average[np.where(amplitude_range == 0)])
            plt.ylabel('Beam $\\frac{1}{e^2} $ waist / mm')
            plt.xlabel('Amplitude / Milliwaves')
            plt.title('Beam Waists for Measurement '+ str(measure_value) + ' beam number ' + str(beam_no+1))
            plt.legend()

            # #plotting for intensity
            # plt.figure()
            # plt.errorbar(amplitude_range*1000, x_max_intensity_average[start_range:end_range], yerr=int_x_err_quad[start_range:end_range], marker='o', linestyle='', label='Mean beam waist in x for beam' +str(beam_no+1), color='blue')
            # plt.errorbar(amplitude_range*1000, y_max_intensity_average[start_range:end_range], yerr=int_y_err_quad[start_range:end_range], marker='o', linestyle='',  label= 'Mean beam waist in y', color='orange')

            # plt.ylabel('Integrated Intensity (px)')
            # plt.xlabel('Amplitude / Milliwaves')
            # plt.title('Intensity for Measurement '+ str(measure_value) + ' beam number ' + str(beam_no+1))
            # plt.legend()

            print(wx_average)
            print(wy_average)
            print(wx_err_quad)

        plt.show()

if __name__ == "__main__":
    main()


