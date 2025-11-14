import os
import pandas as pd
import numpy as np
import pcntoolkit as ptk 
from pcntoolkit.util.utils import create_design_matrix
from pcntoolkit.dataio.fileio import save as ptksave

# globals
root_dir = '/project/3022054.01/projects/linschl/results/Run_8'
out_dir = '/project/3022054.01/projects/linschl/results/Run_8_inv'
data_dir = '/project/3022054.01/projects/linschl/data'
mask_nii = ('/project/3022054.01/projects/linschl/data_nathalie/GM_resample.nii')
ex_nii = os.path.join(data_dir,'hbs', 'hbs_4D_log_gm.nii.gz')

proc_dir = os.path.join(root_dir)

# load covariates
print('loading covariate data ...')
df_cov= pd.read_csv(os.path.join(data_dir,'full_metadata_rescaled_20241210.csv'))
df_cov['age'].describe()
# min 18
# max 65

#%% load tr and te files  
df_te = pd.read_csv(os.path.join(root_dir, 'metadata_tr.csv')) # switch train and test 
df_tr = pd.read_csv(os.path.join(root_dir, 'metadata_te.csv'))

# get indices
tr = df_cov['subj_id'].isin(df_tr['subj_id'])
te = df_cov['subj_id'].isin(df_te['subj_id'])


# extract a list of unique site ids from the training set
site_ids =  sorted(set(df_tr['site'].to_list()))

df_tr.to_csv(os.path.join(out_dir,'metadata_tr.csv'))
df_te.to_csv(os.path.join(out_dir,'metadata_te.csv'))


#%% Configure covariates

# design matrix parameters
df_tr['age'].describe()

xmin = 14 #REAL: 19 # boundaries for ages of participants +/- 5
xmax = 79 #REAL:74

cols_cov = ['age','sex', 'emotional_neglect_scaled','emotional_abuse_scaled', 'physical_abuse_scaled', 'sexual_abuse_scaled','eTIV'] # subscale model
#cols_cov = ['age','sex', 'CTQ_sum','eTIV'] # aggregate model

print('configuring covariates ...')
X_tr = create_design_matrix(df_tr[cols_cov],
                            site_ids = df_tr['site'],
                            basis = 'bspline',
                            #basis_column=0,
                            xmin = xmin,
                            xmax = xmax)

X_te = create_design_matrix(df_te[cols_cov],
                            site_ids = df_te['site'],
                            all_sites=site_ids,
                            basis = 'bspline',
                            #basis_column = 0,
                            xmin = xmin,
                            xmax = xmax)


cov_file_tr = os.path.join(out_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(out_dir, 'cov_bspline_te.txt')
ptk.dataio.fileio.save(X_tr, cov_file_tr)
ptk.dataio.fileio.save(X_te, cov_file_te)

#%% configure response data

data_nii = []
data_nii.append(os.path.join(data_dir, 'mindset','mindset_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'hbs', 'hbs_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'imagen','imagen_4D_log_gm.nii.gz'))
#data_nii.append(os.path.join(data_dir, 'imagen','ctq_sum_score/imagen_ctqsum_4D_log.nii.gz'))
data_nii.append(os.path.join(data_dir, 'become', 'become_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'stratify', 'stratify_4D_log_gm.nii.gz'))


x_te = ptk.dataio.fileio.load(os.path.join(root_dir,'resp_tr.nii.gz'), mask=mask_nii, vol=False).T
x_tr = ptk.dataio.fileio.load(os.path.join(root_dir,'resp_te.nii.gz'), mask=mask_nii, vol=False).T


x_te[np.isnan(x_te)] = 0
x_tr[np.isnan(x_tr)] = 0


# and write out as pkl
resp_file_tr = os.path.join(out_dir,'resp_tr.pkl')
resp_file_te = os.path.join(out_dir,'resp_te.pkl')
ptk.dataio.fileio.save(x_tr, resp_file_tr)
ptk.dataio.fileio.save(x_te, resp_file_te)

# save as nifti
ptksave(x_tr, os.path.join(out_dir,'resp_tr.nii.gz'), example=ex_nii, mask=mask_nii) #,dtype='uint32')
ptksave(x_te, os.path.join(out_dir,'resp_te.nii.gz'), example=ex_nii, mask=mask_nii) #,dtype='uint32')


