import os
import numpy as np
import pandas as pd
from pcntoolkit.util.utils import calibration_descriptives
from pcntoolkit.dataio.fileio import load_nifti, save_nifti
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave

root_dir = '/project/3022054.01/projects/linschl/results/Run_8_inv/'
data_dir = '/project/3022054.01/projects/linschl/data'
mask_nii = ('/project/3022054.01/projects/linschl/data/GM_resample.nii')
ex_nii = os.path.join(data_dir,'hbs', 'hbs_4D_log_gm.nii.gz')  # example file to match header information

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')

output_suffix = '_ref'

#batch_size = 400 


#######################################
# Load metrics
######################################

y_te = ptkload(os.path.join(proc_dir,'resp_te.pkl')) 
yhat = ptkload(os.path.join(w_dir,'yhat_estimate.pkl')) 
Z = ptkload(os.path.join(w_dir,'Z_estimate.pkl'))
EV = ptkload(os.path.join(w_dir,'EXPV_estimate.pkl'))
rho = ptkload(os.path.join(w_dir,'Rho_estimate.pkl'))
smse = ptkload(os.path.join(w_dir,'SMSE_estimate.pkl'))

Z[np.isnan(Z)] = 0
Z[np.isinf(Z)] = 0

[skew, sds, kurtosis, sdk, semean, sesd] = calibration_descriptives(Z) 

# extract voxels with bad kurtosis and skew
badk = np.abs(kurtosis) > 10
kurtosis2 = kurtosis[~badk]

bads = np.abs(skew) > 10
skew2 = skew[~bads]

# describe evaluation metrics
print(sum(badk))
print(kurtosis.max())

print(sum(bads))
print(skew.max())
print(skew.min())


# fix some random bad voxels 
EV[EV < -1] = 0

# save files as nifti
ptksave(rho.T,os.path.join(w_dir,'rho' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(skew, os.path.join(w_dir,'skew' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(kurtosis, os.path.join(w_dir,'kurtosis' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(EV.T, os.path.join(w_dir,'EV' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(smse.T, os.path.join(w_dir,'SMSE' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(Z, os.path.join(w_dir,'Z' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(y_te, os.path.join(w_dir,'y' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(yhat, os.path.join(w_dir,'yhat' + output_suffix + '.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(np.arange(len(EV)), os.path.join(w_dir,'idx.nii.gz'), example=ex_nii, mask=mask_nii, dtype='uint32') #index file as reference 


# load the index again in volumetric form
idxvol = ptkload(os.path.join(w_dir,'idx.nii.gz'), mask=mask_nii, vol=True)

# find the voxel coordinates for a given value
vox_id = np.where(bads)[0][0]
vox_coord = np.asarray(np.where(idxvol == vox_id)).T

# alternative method (opposite direction)
vox_coord = (35,57,31)
vox_id = int(idxvol[vox_coord])

# find batch id
#batch_num, mod_num = divmod(vox_id, batch_size)
#batch_num = batch_num + 1 # batch indexing starts at 1


