#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:01:32 2024

@author: linschl
"""
import os
import pickle
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import pcntoolkit as ptk 
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit.dataio.fileio import save_nifti, load_nifti
from pcntoolkit.normative import predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives

from scipy import stats


#%% globals
root_dir = '/project/3022054.01/projects/linschl/results/Run_8/'
data_dir = os.path.join(root_dir)
w_dir = os.path.join(root_dir,'vox/')
out_dir = os.path.join(root_dir,'vox/structure_coefficients/')
out_dir_samples = os.path.join(out_dir,'per_sample/')
out_dir_group = os.path.join(out_dir,'per_group/')

#%% load covariate data and response file

mask_nii = ('/project/3022054.01/projects/linschl/data/GM_resample.nii')
yhat_est = ptkload(os.path.join(w_dir,'yhat_estimate.pkl'), mask=mask_nii)
df_dem = pd.read_csv(os.path.join(data_dir,'metadata_te_cidi_v2.csv'))

#%% create dummy variables for site and define covariates
dummies = pd.get_dummies(df_dem['site'])
dummies.columns = [f'site{i+1}' for i in range(dummies.shape[1])]
df_dem_dummy = pd.concat([df_dem, dummies], axis=1)

cols_cov = ['age','sex', 'emotional_neglect_scaled','emotional_abuse_scaled', 'physical_abuse_scaled', 'sexual_abuse_scaled','eTIV', 'site1', 'site2', 'site3', 'site4', 'site5', 'site6','site7', 'site8','site9','site10' ] #subscale model
#cols_cov = ['age','sex', 'CTQ_sum','eTIV', 'site1', 'site2', 'site3', 'site4', 'site5', 'site6','site7', 'site8','site9' ] #aggregate model


#%% LOOP THROUGH ALL COVARIATES:

yhat = pd.DataFrame(yhat_est)  
for column in cols_cov:
    # Calculate the correlation between the current covariate and all voxels
    rho = yhat.corrwith(df_dem_dummy[column])
    
    # Convert the result to a NumPy array and reshape if needed
    rho_array = rho.to_numpy()[:, np.newaxis]
    
    # compute Rho squared    
    rho_sq = rho_array * rho_array
    
    # Build the output filename based on the current column name
    filename = os.path.join(out_dir, f'Rho_{column}.nii.gz')
    filename_sq = os.path.join(out_dir, f'RhoSquared_{column}.nii.gz')
    
    # Save the correlation result as a NIfTI file
    save_nifti(rho_array, filename, mask=mask_nii, examplenii=mask_nii, dtype='float32')
    save_nifti(rho_sq, filename_sq, mask=mask_nii, examplenii=mask_nii, dtype='float32')   

#%% sum site Rho_squared
nifti_files = glob.glob(os.path.join(out_dir, 'RhoSquared_site*')) 

sum_data = None

for file in nifti_files:
    nifti = ptk.dataio.fileio.load(file, mask=mask_nii, vol=False).T
    
    #data = nifti.get_fdata()
    
    if sum_data is None:
        sum_data = np.zeros_like(nifti)
    
    sum_data += nifti

# Save the new NIfTI image
filename = os.path.join(out_dir, 'RhoSquared_site.nii.gz')
save_nifti(sum_data, filename, mask=mask_nii, examplenii=mask_nii, dtype='float32')




#%% calculate adversity structure coefficients for each sample


# get index per site
is_mindset = df_dem['dataset'] == "MINDSet"
mindset_indices = list(df_dem.index[is_mindset])

is_hbs = df_dem['dataset'] == "HBS"
hbs_indices = list(df_dem.index[is_hbs])

is_become = df_dem['dataset'] == "BECOME"
become_indices = list(df_dem.index[is_become])

is_img = df_dem["dataset"] == "IMAGEN"
img_indices = list(df_dem.index[is_img])

is_str = df_dem["dataset"] == "STRATIFY"
str_indices = list(df_dem.index[is_str])


# define adversity columns
cols_adv = ['emotional_neglect_scaled','emotional_abuse_scaled', 'physical_abuse_scaled', 'sexual_abuse_scaled']
#cols_adv = ['CTQ_sum']

# MINDSET
df_yhat = yhat.iloc[mindset_indices, :]
df_cov = df_dem.iloc[mindset_indices,:]

for column in cols_adv:
    # Calculate the correlation between the current covariate and all voxels
    rho = df_yhat.corrwith(df_cov[column])
    
    # Convert the result to a NumPy array and reshape if needed
    rho_array = rho.to_numpy()[:, np.newaxis]
    
    # compute Rho squared    
    #rho_sq = rho_array * rho_array
    
    # Build the output filename based on the current column name
    filename = os.path.join(out_dir_samples, f'Rho_{column}_mindset.nii.gz')
    #filename_sq = os.path.join(out_dir_samples, f'RhoSquared_{column}_mindset.nii.gz')
    
    # Save the correlation result as a NIfTI file
    save_nifti(rho_array, filename, mask=mask_nii, examplenii=mask_nii, dtype='float32')
    #save_nifti(rho_sq, filename_sq, mask=mask_nii, examplenii=mask_nii, dtype='float32')   

#%% calculate adversity structure coefficients for each group
yhat = pd.DataFrame(yhat_est) 

is_patient= df_dem['diagnosis'] == 1
pat_indices = list(df_dem.index[is_patient])

is_control= df_dem['diagnosis'] == 0
con_indices = list(df_dem.index[is_control])

# define adversity columns
cols_adv = ['emotional_neglect_scaled','emotional_abuse_scaled', 'physical_abuse_scaled', 'sexual_abuse_scaled']
#cols_adv = ['CTQ_sum']

# MINDSET
df_yhat = yhat.iloc[con_indices, :]
df_cov = df_dem.iloc[con_indices,:]

for column in cols_adv:
    # Calculate the correlation between the current covariate and all voxels
    rho = df_yhat.corrwith(df_cov[column])
    
    # Convert the result to a NumPy array and reshape if needed
    rho_array = rho.to_numpy()[:, np.newaxis]
    
    # compute Rho squared    
    #rho_sq = rho_array * rho_array
    
    # Build the output filename based on the current column name
    filename = os.path.join(out_dir_group, f'Rho_{column}_controls.nii.gz')
    #filename_sq = os.path.join(out_dir_samples, f'RhoSquared_{column}_mindset.nii.gz')
    
    # Save the correlation result as a NIfTI file
    save_nifti(rho_array, filename, mask=mask_nii, examplenii=mask_nii, dtype='float32')
    #save_nifti(rho_sq, filename_sq, mask=mask_nii, examplenii=mask_nii, dtype='float32')   



    