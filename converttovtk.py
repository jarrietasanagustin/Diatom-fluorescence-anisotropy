"""
Programa para convertir imagenes en formato .tiff a formato .vtk
El programa lee las imágenes, se define un sistema de coordenadas y
y los datos se guardan en un archivo .vtk en formato de puntos.

Hay dos cosas que se deben hacer:
1- Automatizar la obtención de las coordenadas x_center y y_center
2- Recortar las imágenes para que ocupen menos espacio en memoria

Jorge Arrieta Marzo 2024
"""

import numpy as np
import skimage as ski
import cv2 
from scipy import ndimage as ndi
import os
import sys  
import glob
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
from pyevtk.hl import pointsToVTK
from pyevtk.hl import gridToVTK 
import pyvista
#Ruta de las imágenes
path='/Users/jorge/Documents/Research/Diatoms_fluorescence/FL/'
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
#las imagenes se guardan en un arreglo de numpy de 3 dimensiones


x_coord=np.zeros(np.shape(imagen_kk)[0],dtype=np.float64)
y_coord=np.zeros(np.shape(imagen_kk)[0],dtype=np.float64)
col_min=2150;
col_max=2650;
row_min=1400;
row_max=1900;
img_stack=np.zeros((number_images-1,500,500),dtype=np.int64)
#img_stack=np.zeros((36,500,500),dtype=np.int64)
#images=np.zeros(number_images,);
for ii in range(0,number_images-1):
     #images=cv2.imread(path+lst[index_sort[ii]],0);
     images=ski.io.imread(path+lst[index_sort[ii]])
     img_crop=images[row_min:row_max,col_min:col_max]
#     images = np.array(images);
     img_stack[ii,:,:]=img_crop


y_center=250#p.shape(images)[0]/2;
x_center=210#this value is obtained by direct observation of the images
#needs to be automated;
x_coord=np.linspace(0,np.shape(img_crop)[1]-1,np.shape(img_crop)[1])-x_center;
y_coord=np.linspace(0,np.shape(img_crop)[0]-1,np.shape(img_crop)[0])-y_center;
r=np.linspace(0,249,250);
y_plano=np.zeros(np.shape(y_coord)[0],dtype=np.float64)
z_plano=np.zeros(np.shape(y_coord)[0],dtype=np.float64)
X, Y=np.meshgrid(x_coord,y_coord,indexing='xy');

X_1=np.zeros((number_images-1,500,500),dtype=np.float64)
Y_1=np.zeros((number_images-1,500,500),dtype=np.float64)
Z_1=np.zeros((number_images-1,500,500),dtype=np.float64)

#X_1=np.zeros((36,500,500),dtype=np.float64)
#Y_1=np.zeros((36,500,500),dtype=np.float64)
#Z_1=np.zeros((36,500,500),dtype=np.float64)


theta = np.linspace(1e-3, 2*np.pi, number_images-1)
for ii in range(0,number_images-1):
 #    y_plano[0:250]=r*np.cos(theta[ii]);
#     y_plano[250:500]=r*np.cos(theta[ii]+np.pi);
     y_plano=y_coord*np.cos(theta[ii]);
     #z_plano[0:250]=r*np.sin(theta[ii]);
     #z_plano[250:500]=r*np.sin(theta[ii]+np.pi);
     z_plano=y_coord*np.sin(theta[ii]);
     x_plano=x_coord;
     Z_plano, Y_plano=np.meshgrid(z_plano,y_plano,indexing='xy');
     X_plano, Y_plano_1=np.meshgrid(x_plano,y_plano,indexing='xy');
    # X_1[ii,:,:]=X_plano;
    # Y_1[ii,:,:]=Y_plano;
#    · Z_1[ii,:,:]=Z_plano;
     X_1[ii,:,:]=X;
     Y_1[ii,:,:]=Y*np.cos(theta[ii]);
     Z_1[ii,:,:]=Y*np.sin(theta[ii]);
    

#     imagenes=np.array([X,Y_plano,Z_plano,img_stack[:,:,ii]]);
x_1=np.reshape(X_1,np.size(X_1,0)*np.size(X_1,1)*np.size(X_1,2),order='F')
y_1=np.reshape(Y_1,np.size(X_1,0)*np.size(X_1,1)*np.size(X_1,2),order='F')
z_1=np.reshape(Z_1,np.size(X_1,0)*np.size(X_1,1)*np.size(X_1,2),order='F')
im_1=np.reshape(img_stack,np.size(X_1,0)*np.size(X_1,1)*np.size(X_1,2),order='F')
plt.contourf(X_1[45,:,:],Z_1[45,:,:],img_stack[45,:,:],cmap='viridis')
plt.colorbar()
pointsToVTK("./image", x_1, y_1, z_1, data = {"fluor" : im_1} )
gridToVTK("./prueba", X_1, Y_1, Z_1, pointData = {"fluor" : img_stack} )


points=np.transpose(np.array( [x_1,y_1,z_1]))
mesh = pyvista.PolyData(points)
alpha = 17;
volume = mesh.delaunay_3d(alpha=alpha)
ore = volume.threshold(1200)
p = pyvista.Plotter(shape=(1,3), notebook=False)
p.subplot(0,0)
p.add_mesh(points, point_size=5, render_points_as_spheres=True)
p.show_grid()
p.subplot(0,1)
p.add_mesh(volume)
p.show_grid()
