# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:28:16 2018

@author: John
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 
import pathlib
import pydicom
from itertools import chain
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

ds=pydicom.dcmread('c:\\Users\\Public\\PhtmTest.dcm')
image=ds.pixel_array
pspx,pspy=ds.PixelSpacing
rescale_intercept=int(ds.RescaleIntercept)
threshold=threshold_otsu(image)
bw=clear_border(morphology.closing(image>threshold,morphology.square(3)))
plt.figure(dpi=300)
plt.title('image')
plt.imshow(image,cmap=plt.cm.bone)
plt.show()
plt.figure(dpi=300)
plt.title('bw')
plt.imshow(bw)
plt.show()
labelimage=label(bw,connectivity =1)
props=regionprops(labelimage)
area=[ele.area for ele in props ]
largest_blob_indx=np.argmax(area)
largest_blob_label=props[largest_blob_indx].label
body_pixels=np.zeros(image.shape,dtype=np.uint8)
body_pixels[labelimage == largest_blob_label] = 1
plt.figure(dpi=300)
plt.title('body pixels')
plt.imshow(body_pixels)
plt.show()
filled_body_pixels=ndi.binary_fill_holes(body_pixels)
plt.figure(dpi=300)
plt.title('filled body pixels')
plt.imshow(filled_body_pixels)
plt.show()
pixel_count=np.sum(filled_body_pixels)
pixel_radius=math.sqrt(pixel_count/math.pi)
eq_diam=.2*pspx*pixel_radius
pixel_attenuations= image[filled_body_pixels==1]
atest2=[(((i+rescale_intercept)/1000. )+1.0) for i in np.nditer(pixel_attenuations)]
tot_pix_area=sum(atest2)
pa_radius=math.sqrt(tot_pix_area/math.pi)
pa_diam=.2*pa_radius*pspx



