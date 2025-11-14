import os
import pandas as pd
import numpy as np
import pcntoolkit as ptk 
from pcntoolkit.util.utils import create_design_matrix
from pcntoolkit.dataio.fileio import save as ptksave

# globals
root_dir = '/project/3022000.05/linschl/results/Run_9'
data_dir = '/project/3022000.05/linschl/data'
mask_nii = ('/project/3022000.05/linschl/data_nathalie/GM_resample.nii')
ex_nii = os.path.join(data_dir,'hbs', 'hbs_4D_log_gm.nii.gz')

proc_dir = os.path.join(root_dir)

# load covariates
print('loading covariate data ...')
df_cov= pd.read_csv(os.path.join(data_dir,'full_metadata_ctq_sum_score_20241219.csv'))
df_cov['age'].describe()
# min 18
# max 65

#%% HOLD OUT STRATIFY
df_str = df_cov[df_cov['dataset'] == 'STRATIFY']
df_cov = df_cov[~(df_cov['dataset'] == 'STRATIFY')]

df_c = df_str[df_str['diagnosis'] == 0]
df_p = df_str[~(df_str['diagnosis'] == 0)]

#%% SPLIT HALF TRAINING TEST

tr = np.random.uniform(size=df_cov.shape[0]) > 0.5
te = ~tr

df_tr = df_cov.iloc[tr]
df_tr.to_csv(os.path.join(proc_dir,'metadata_tr.csv'))
df_te = df_cov.iloc[te]
df_te.to_csv(os.path.join(proc_dir,'metadata_te.csv'))

# add stratify controls to training set
df_tr = pd.concat([df_tr, df_c])

#%% SPLIT STRATIFY

# randomly divide patients and assign 10-15 to the training set
num_tr_p = np.random.randint(10, 16)

# Shuffle the indices 
shuffled_indices = np.random.permutation(df_p.index)

# Split the indices into training and testing
tr_indices = shuffled_indices[:num_tr_p]
te_indices = shuffled_indices[num_tr_p:]


df_tr_str = df_p.loc[tr_indices]
df_te_str = df_p.loc[te_indices]

#%% ADD STRATIFY PATIENTS TO TEST AND TRANING SPLITS
df_tr_all = pd.concat([df_tr, df_tr_str])
df_te_all = pd.concat([df_te, df_te_str])


# AGE
df_te_all['age'].describe()
df_tr_all['age'].describe()

# min 19
# max 63


# extract a list of unique site ids from the training set
site_ids =  sorted(set(df_tr_all['site'].to_list()))

df_tr_all.to_csv(os.path.join(proc_dir,'metadata_tr.csv'))
df_te_all.to_csv(os.path.join(proc_dir,'metadata_te.csv'))


#%% CONFIGURE COVARIATES


# design matrix parameters
xmin = 14 #REAL: 19 # boundaries for ages of participants +/- 5
xmax = 58 #REAL:63

#cols_cov = ['age','sex', 'emotional_neglect_scaled','emotional_abuse_scaled', 'physical_abuse_scaled', 'sexual_abuse_scaled','eTIV'] #subscale model
cols_cov = ['age','sex', 'CTQ_sum','eTIV'] #aggregate model

print('configuring covariates ...')
X_tr = create_design_matrix(df_tr_all[cols_cov],
                            site_ids = df_tr_all['site'],
                            basis = 'bspline',
                            #basis_column=0,
                            xmin = xmin,
                            xmax = xmax)

X_te = create_design_matrix(df_te_all[cols_cov],
                            site_ids = df_te_all['site'],
                            all_sites=site_ids,
                            basis = 'bspline',
                            #basis_column = 0,
                            xmin = xmin,
                            xmax = xmax)


cov_file_tr = os.path.join(proc_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(proc_dir, 'cov_bspline_te.txt')
ptk.dataio.fileio.save(X_tr, cov_file_tr)
ptk.dataio.fileio.save(X_te, cov_file_te)

#%% CONFIGURE RESPONSE DATA

data_nii = []
#data_nii.append(os.path.join(data_dir, 'mindset','mindset_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'hbs', 'hbs_4D_log_gm.nii.gz'))
#data_nii.append(os.path.join(data_dir, 'imagen','imagen_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'imagen','ctq_sum_score/imagen_ctqsum_4D_log.nii.gz'))
data_nii.append(os.path.join(data_dir, 'become', 'become_4D_log_gm.nii.gz'))
data_nii.append(os.path.join(data_dir, 'stratify', 'stratify_4D_log_gm.nii.gz'))


# load the response data as nifti
print('loading wholebrain response data ...') 
for i, f in enumerate(data_nii):
    print('loading study', i, '[', f, '] ...')
    if i == 0:
        x = ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T
        print(x.shape)
    else: 
        x1 = ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T #x1 temporarily holds new data before concatenating with x
        print(x1.shape)
        x = np.concatenate((x, ptk.dataio.fileio.load(f, mask=mask_nii, vol=False).T))
        print(x.shape)

x[np.isnan(x)] = 0


tr_indices = df_tr_all.index.to_list()  # Gets the row indices of df_tr_all
te_indices = df_te_all.index.to_list()  # Gets the row indices of df_te_all


# write out as pkl
resp_file_tr = os.path.join(proc_dir,'resp_tr.pkl')
resp_file_te = os.path.join(proc_dir,'resp_te.pkl')
ptk.dataio.fileio.save(x[tr_indices,:], resp_file_tr)
ptk.dataio.fileio.save(x[te_indices,:], resp_file_te)

# save as nifti
ptksave(x[tr_indices,:], os.path.join(proc_dir,'resp_tr.nii.gz'), example=ex_nii, mask=mask_nii) #,dtype='uint32')
ptksave(x[te_indices,:], os.path.join(proc_dir,'resp_te.nii.gz'), example=ex_nii, mask=mask_nii) #,dtype='uint32')


