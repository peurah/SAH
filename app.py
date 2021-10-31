# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:08:31 2021

@author: hus56590733
"""

import streamlit as st
import nibabel as nib
from io import BytesIO
import numpy as np
import os
import tempfile
from gzip import GzipFile
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

HU_min = 0.
HU_max = 100.
    
def normalize(image):
    image = (image - HU_min) / (HU_max - HU_min)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def plot_slice1(vol, slice_ix):
    fig, ax = plt.subplots()
    plt.axis('off')
    selected_slice = vol[:,:,slice_ix]
    ax.imshow(selected_slice, origin='lower', cmap='gray')
    return fig

def plot_slice2(vol,slice_ix):
    fig, ax = plt.subplots()
    plt.axis('off')
    selected_slice = vol[slice_ix,:,:,:]
    ax.imshow(selected_slice)
    return fig

def check_orientation(ct_image, ct_arr):
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
    if y != 'P':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
    if z != 'I':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
    return ct_arr

uploaded_file = st.file_uploader("Choose a image file", type=".nii.gz")

if uploaded_file is not None:
    fh = nib.FileHolder(fileobj=GzipFile(fileobj=BytesIO(uploaded_file.read())))
    nifti = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    os.makedirs('nifti/original',exist_ok=True)
    os.makedirs('nifti/preprocessed',exist_ok=True)
    nib.save(nifti,'nifti/original/original.nii.gz')
    img = np.array(nifti.dataobj)
    img = check_orientation(nifti, img)
    img = np.flipud(img)
    st.title(img.shape)
    predict = st.button('Preprocess, run the algorithm and show results')
    if predict:
        os.system('python preprocess.py nifti/original nifti/preprocessed')
        os.system('python run.py --task inference --config job/config.ini --job-dir job')
        st.title('ready, showing the results')
        try:
            os.remove('job/dataset_split.csv')
        except:
            pass
        try:
            os.remove('job/img.csv')
        except:
            pass
        try:
            os.remove('inferred.csv')
        except:
            pass
        if predict:
            st.title('here1')
            img_nif = nib.load('nifti/preprocessed/original.nii.gz')
            img = np.array(img_nif.dataobj)
            img = check_orientation(img_nif, img)
            img = np.rollaxis(img,2,0)
            img = np.expand_dims(img,3)
            st.title(img.shape)
            img = np.flipud(img)
            
            seg_nif = nib.load('window_seg_original__niftynet_out.nii.gz')
            seg = np.array(seg_nif.dataobj)
            seg = check_orientation(seg_nif, seg)
            seg = np.rollaxis(seg,2,0)
            seg = np.expand_dims(seg,3)[:,:,:,:,0,0]
            st.title(seg.shape)
            seg = np.flipud(seg)
            
            empty = []
            for i in range(len(seg)):
                segmentation = seg[i]
                segmentation = cv2.merge([segmentation,segmentation,segmentation])
                segmentation = np.expand_dims(segmentation,0)
                empty.append(segmentation)
            segfinal = np.concatenate(empty)
            
            empty = []
            for i in range(len(segfinal)):
                x = segfinal[i]
                x1 = np.where(x==1, [255,0,0],0)
                empty.append(np.expand_dims(x1,0))
            test = np.concatenate(empty)

            x = normalize(img)
            x = x*255
            
            empty = []
            for i in range(len(x)):
                one = x[i]
                one = cv2.merge([one,one,one])
                one = cv2.add(one.astype(np.float32),test[i].astype(np.float32))
                one = np.expand_dims(one,0)
                empty.append(one)
            final = np.concatenate(empty)
            
            for i in range(len(final)):
                st.image(np.rot90(final[i,:,:,:],-1,(0,1))/255,clamp=True,width=850)
                
            
            
        
            
            

            
            

            
            
            
            

