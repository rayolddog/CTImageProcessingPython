# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:52:29 2020

@author: JBlaptop1

Preprocess based on Hounsfield units

This was originally directed at preprocessing DICOM files from CT chest scans 
to address the diagnosis of pulmoray embolism.  There is a significant variability 
of radiodensity of injected contrast.  Pulmonary emboli are areas of decreased 
density in the contrast present in pulmonary arteries.  When I read the CT scans,
I use different window settings to look for emboli.  By comparison, looking for 
intracraial hemorrhage can be done mainly with a single window setting (so called 
subdural windows, which have been used since the early 1980s).

Because of the variability of contrast density, I thought it would be better to 
map the Hounsfield units of relative radiodensity using a colormap to preserve the 
11 to 12 bits of density information represented in the Hounsfield units.  I could 
have used the Intensity Hue Saturation transorm as I previously used for colorizing 
CT scans similar to pictures in anatomy textbooks (https://doi.org/10.1007/BF03167750).
However, I wanted to skip the transformation to HSV and use a colormap directly.

I never was competitive in the Kaggle competitions.  I would always get bogged down
in trying to clean up the data.  However, you may useful some of the code for 
recapturing the Houndsfield units from the DICOM files.  I never felt comfortable 
simply using common image normalization techniques to 8bit RGB images. IMHO,
such normalization contributes to variability in the neural networks models'
performance when applied to different manufactures/reconstruction algorithms. 

Of course, recapturing the original Hounsfield units does not solve the problem of
the variability affecting DNN performance.  For that, I would focus on the 14bit
sinogram (raw CT data before reconstruction).

This script reads the DICOM files from the local folder programmed into the script 
and preprocesses them to create a set of images that can be used for training.  

Commented code is included so that the colorized images can be are saved in a directory 
structure that is compatible with the ImageDataGenerator.

The pydicom package cannot read some files pixel data.  As far as I can
tell, it fails on JPEG lossless encoding.  I haven't verified the reason
some files can't be processed.  If the CT slice has very little area of 
aerated pixels, it doesn't process the file.  The preprocessing would have
to be included in the testing of datasets.  If there are too few aerated
pixels, it will assume that a PE isn't present.  
"""

import os
#import pandas as pd
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
import cupy as cp
from skimage.measure import label, regionprops

from skimage import img_as_ubyte
import time

from skimage.io import imsave


CTJPG_dir = "C:\\your directory\\"



def show_image(title_failed,failed_image):
    plt.figure()
    plt.title(title_failed)
    plt.imshow(failed_image)
    plt.show()
    return

PathDicom = CTJPG_dir #"D:\\AIPE\\train\\"
#df=pd.read_csv('C:\\your directory\\train.csv',index_col=2)

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if filename.lower().endswith('.dcm'):  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
            
print('number of dicom files',len(lstFilesDCM))

list_props = []
count_failed_read = 0
count_success = 0
count_too_small_lung_pixels = 0
count_area = 0
count_too_few_body_pixels = 0
count_failed_pixel_encoding = 0
saved_PE = 0
saved_Neg = 0
count_pe_too_small=0

test_range = range(len(lstFilesDCM))

threshold_body = -200
threshold_aerated = -300
threshold_alveoli = -990
start = time.time()
for i in test_range:
    dcmfile_name = lstFilesDCM[i]

    try:
        testfile=pydicom.read_file(dcmfile_name)
    except:
        count_failed_read += 1
        continue
    
    try:
        ctimage=testfile.pixel_array
    except:
        print('Reading dicom image pixel data failed')
        # print(testfile)
        count_failed_pixel_encoding += 1
    else:
        rsz_ct = resize(ctimage,(224,224),preserve_range = True).astype(int)
        resize_ct = cp.array(rsz_ct)
        sop_instance=testfile.SOPInstanceUID
        rescale_slope = float(testfile.RescaleSlope)
        rescale_intercept = int(testfile.RescaleIntercept)
        if rescale_slope != 1.0:
            print('Rescale slope other than 1 has not been tested')
            resize_ct = resize_ct * rescale_slope
            
        if rescale_intercept != 0 :
            resize_ct = resize_ct + rescale_intercept
        resize_ct = resize_ct.astype(int)
        labels=label(cp.asnumpy(resize_ct)> threshold_body )
        props=regionprops(labels)
        if len(props)==0:
            print('ERROR in CT pixel values, none greater than 500, index= ', i)
  
        else:
            list_areas=[x.area for x in props]
            max_area=max(list_areas)
            if max_area <= 2000 :
                try :
                    labels=label(resize_ct > threshold_body)
                    props = regionprops(labels)
                    list_areas = [x.area for x in props]
                    max_area = max(list_areas)
                except :
                    print('error in label function',i)
                    show_image('Failed',cp.asnumpy(resize_ct))
                
            if max_area > 2000:
                build_body_mask = cp.zeros((224,224),dtype = int)
                index_body=list_areas.index(max_area)
                body_mask=props[index_body].filled_image
                x1,y1,x2,y2=props[index_body].bbox
                for index_row_body_mask in range(x2 - x1) :
                    shifted_row_index = index_row_body_mask + x1
                    for index_col_body_mask in range(y2-y1) :
                        shifted_col_index = index_col_body_mask + y1
                        replacement_value = body_mask[index_row_body_mask,index_col_body_mask]
                        build_body_mask[shifted_row_index,shifted_col_index] = replacement_value
                masked_ct=resize_ct*build_body_mask
                aerated_pixels=masked_ct < threshold_aerated
                alveoli=aerated_pixels*masked_ct*build_body_mask * -1
                air_mask=alveoli>0
                count_air_pixels=np.sum(air_mask)

                if count_air_pixels > 1000:
                    max_alveoli = cp.amax(alveoli)
                    scaled_alveoli = alveoli / float(max_alveoli)
                    lungs = cp.asnumpy(scaled_alveoli)
                    focus_mask_bottom = masked_ct > 15                                                                                              
                    focus_mask_top = masked_ct > 300
                    masked_ct[masked_ct > 500] = 500
                    focus = (focus_mask_bottom*masked_ct)
                    upper_focus_mask = focus > 400
                    upper_focus = (focus * upper_focus_mask) - 400
                    upper_focus[(upper_focus < 0)] = 0
                    focus[(focus > 400)] = 400
                    np_focus = cp.asnumpy(focus)
                    np_upper_focus = cp.asnumpy(upper_focus)
                    pix_min = np.amin(np_focus)
                    if pix_min < 0 :
                        np_focus = np_focus - pix_min
                    pix_max = np.amax(np_focus)
                    focus_interest = np_focus/pix_max
                    pix2_min = np.amin(np_upper_focus)
                    if pix2_min < 0:
                        np_upper_focus = np_upper_focus - pix2_min
                    pix2_max = np.amax(np_upper_focus)
                    upper_focus = np_upper_focus / pix2_max
                    dims = np_focus.shape
                    rows_img = dims[0]
                    cols_img = dims[1]
                    
                    u_lungs = img_as_ubyte(lungs)
                    u_focus = img_as_ubyte(focus_interest)
                    u_upper = img_as_ubyte(upper_focus)
                    
                    color_img2 = np.zeros((rows_img,cols_img,3),dtype='uint8')
                    color_img2[:,:,0] = u_focus
                    color_img2[:,:,1] = u_lungs
                    color_img2[:,:,2] = u_upper 
 
                    # show_image('Coded image based on Hounsfield values',color_img2)
                    count_success += 1
                    show_image('Color',color_img2)
# =============================================================================
#                     dir1,filename = os.path.split(dcmfile_name)
#                     dir2,dir3 = os.path.split(dir1)
#                     jpg_filename = filename.replace('.dcm','.jpg')
#                     add_dirname = dir2.replace('D:\\AIPE\\train','C:\\AIPE\\CTPEjpg\\Mix') 
#                     if not os.path.isdir(add_dirname):
#                         os.makedirs(add_dirname)
# =============================================================================
# =============================================================================
#      
#                     if pe_present :
#                         filename_out =add_dirname + '\\PE' + jpg_filename 
#                         if not os.path.isfile(filename_out):
#                             imsave(filename_out,color_img2)
#                             saved_PE += 1
#                         else:
#                             print('Clash filenames',filename_out)
#                     else :
#                         filename_out = add_dirname + '\\NPE' + jpg_filename
#                         if not os.path.isfile(filename_out):
#                             imsave(filename_out, color_img2)
#                             saved_Neg += 1
#                         else: 
#                             print('Clash filenames',filename_out)
#                     
#                 else :
#                     if pe_present :
#                         show_image('Small pos for PE', cp.asnumpy(resize_ct))
#                         count_pe_too_small += 1
#                     count_too_small_lung_pixels += 1
# =============================================================================
            else:
                count_too_few_body_pixels += 1
                show_image('Failed count',cp.asnumpy(resize_ct))
            
                        
finish = time.time()
elapsed_time = finish - start
print('Elapsed ', elapsed_time)
print('Successes',count_success)
print('Too small lung pixels',count_too_small_lung_pixels)
print('Failed body pixel count', count_too_few_body_pixels)
print('Number saved with PE', saved_PE)
print('Number saved without PE',saved_Neg)
                
