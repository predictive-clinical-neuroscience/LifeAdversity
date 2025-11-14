import os
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
from pcntoolkit.util.utils import calibration_descriptives
from pcntoolkit.dataio.fileio import load_nifti, save_nifti
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import statsmodels.api as sm

root_dir = '/project/3022054.01/projects/linschl/results/Run_8/'
data_dir = '/project/3022054.01/projects/linschl/data'
mask_nii = ('/project/3022054.01/projects/linschl/data/GM_resample.nii')
ex_nii = os.path.join(data_dir,'hbs', 'hbs_4D_log_gm.nii.gz')  # example file to match header information

proc_dir = os.path.join(root_dir)
w_dir = os.path.join(proc_dir,'vox/')

# In[ ]: Load metrics

EV = ptkload(os.path.join(w_dir,'EXPV_estimate.pkl'))
smse = ptkload(os.path.join(w_dir,'SMSE_estimate.pkl'))
Z = ptkload(os.path.join(w_dir,'Z_estimate.pkl'))
Z[np.isnan(Z)] = 0
Z[np.isinf(Z)] = 0
[skew, sds, kurtosis, sdk, semean, sesd] = calibration_descriptives(Z) 



# In[ ]: Histogram of model evaluation metrics 

# EV histogram
sns.set(style='white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(EV)+0.005)

ax.hist(EV, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Explained Variance')
plt.ylabel('Number of voxels')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
#plt.show()
fig.savefig(os.path.join(root_dir, 'figures','EV_histogram.png'), dpi=300)


# In[ ]: Skew histogram
sns.set(style='white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(skew)+0.005)
min_x = (min(skew)-0.005)

ax.hist(skew, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Skew')
plt.ylabel('Number of voxels')
plt.axis([min_x,max_x, 0,60000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)

#plt.show()
plt.tight_layout()
fig.savefig(os.path.join(root_dir, 'figures','Skew_histogram.png'), dpi=300)

# In[ ]:  Kurtosis histogram
data = kurtosis

sns.set(style='white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(data)+0.005)
min_x = (min(data)-0.005)

ax.hist(data, bins = 800, ec = 'white', lw=0.2, fc = '#9D9D9D') 
plt.xlabel('Kurtosis')
plt.ylabel('Number of voxels')
plt.xlim([min_x, 50])
#plt.ylim([0, 160000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)

#plt.show()
plt.tight_layout()
fig.savefig(os.path.join(root_dir, 'figures','Kurtosis_histogram.png'), dpi=300)

# In[ ]: SMSE histogram

data = smse

sns.set(style='white')
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots()

max_x = (max(data)+0.005)
min_x = (min(data)-0.005)

ax.hist(data, bins = 100, ec = 'white', lw=0.2, fc = '#9D9D9D')
#plt.xlim([-0.5, 1.1])
#plt.ylim([0, 13000])
plt.xlabel('SMSE')
plt.ylabel('Number of voxels')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)

#plt.show()
plt.tight_layout()
fig.savefig(os.path.join(root_dir, 'figures','smse_histogram.png'), dpi=300)


