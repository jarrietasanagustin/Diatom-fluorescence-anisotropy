#_______________________________________________________________________________
#This program is used to find the center of the pipette in the image.
#This is done bi locating the peaks of the intensity of the image read by rows
#The the the distance between the peaks is calculated and the center of the pipette
#corresponds with the half of the distance between the peaks of the intensity
#Jorge Arrieta April 2024
#_______________________________________________________________________________


import numpy as np
import skimage as ski
from scipy import ndimage as ndi
from scipy import signal
import os
import sys  
import glob
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
path='/Users/jorge/Documents/Research/Diatoms_fluorescence/BF/'
lst = os.listdir(path)
numbers=np.zeros(len(lst),dtype=np.int64)

for ii in range(0,len(lst)):
     img_name=lst[ii]
     img_number=img_name[9:14]
     pp=img_number.find('_',2)
     if pp==2:
          numbers[ii]=int(img_number[pp-1])
     else:
          numbers[ii]=int(img_number[1:pp])
index_sort=np.argsort(numbers)
number_images = len(lst)
#_______________________________________________________________________
#La lista de imagenes se ordena en función del ángulo de rotación
#para ello se extrae el ángulo de rotación de cada imagen del nombre
#del archivo
#_______________________________________________________________________

imagen_kk=ski.io.imread(path+lst[0])



col_min=2150;
col_max=2900;
row_min=1200;
row_max=2100;
# Crop the image
#cropped_image = imagen_kk[row_min:row_max, col_min:col_max]
img_stack=np.zeros((number_images-1,row_max-row_min,col_max-col_min),dtype=np.float64)

for ii in range(0,number_images-1):
     #images=cv2.imread(path+lst[index_sort[ii]],0);
     images=ski.io.imread(path+lst[index_sort[ii]])
     img_crop=images[row_min:row_max,col_min:col_max]
#     images = np.array(images);
     img_stack[ii,:,:]=img_crop


background = np.mean(img_stack, axis=0)
# Show the cropped image
plt.imshow(img_stack[0,:,:]-background, cmap='gray')
plt.axis('off')
plt.show()

#for ii in range(0,cropped_image.shape[0]):
# Find the peaks of intensity
#    peaks = peak_local_max(cropped_image[ii, :], min_distance=10, threshold_abs=50) 
peaks,_ = signal.find_peaks(cropped_image[:, 700]-background[:,700], prominence=(400)) 
peakind = signal.find_peaks_cwt(cropped_image[:, 700]-background[:,700], 5)#
plt.plot(cropped_image[:,700])
plt.show()

plt.imshow(cropped_image, cmap='gray')
plt.plot([700,700,700,700],peaks, "or")
plt.show()
test=ski.filters.sobel(cropped_image-background)
test_1=ski.feature.canny(cropped_image,sigma=2)
tested_angles = np.linspace(-np.pi*10/180, np.pi*10/180, 10, endpoint=False)
h, theta, d = ski.transform.hough_line(test_1, theta=tested_angles)
# Plot the lines obtained by the Hough transform
fig, ax = plt.subplots()
for _, angle, dist in zip(*ski.transform.hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2)) 
plt.show()



# Print the coordinates of the peaks
