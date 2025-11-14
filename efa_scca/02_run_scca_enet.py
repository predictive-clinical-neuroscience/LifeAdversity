#%% Imports
import os
import sys
import pandas as pd
import numpy as np
import pcntoolkit as ptk 

from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score


if 'utils' in sys.modules:
    del sys.modules['utils']

wdir = '/project/3022000.05/projects/sscca'
sys.path.append(os.path.join(wdir, 'saccade'))

from scca import MSCCA, mscca_fit_predict
from utils import compute_loadings

def fast_r(A,B):
    N = A.shape[0]

    #A = np.tile(Cm.scores[0][:,0], (B.shape[1],1)).T
    #B  = X2tr

    # first mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del (Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)
    return Rho

#%% Set globals

# globals
prj_dir = '/project/3022054.01/projects/linschl/'

datadir = os.path.join(prj_dir,'data','EFA')
Zdir = os.path.join(prj_dir, 'results','Run_8','vox')
Zdir_2 = os.path.join(prj_dir, 'results','Run_8_inv','vox')

mask_nii = os.path.join(prj_dir,'data','GM_resample.nii')
ex_nii = os.path.join(Zdir, 'Z_ref.nii.gz')

dataset = 'ima_str'
rep = 2 # 1 , 2 or None
print(dataset)

out_dir = os.path.join(prj_dir,'results','CCA','Run_6', dataset + '_v2_nopos')
os.makedirs(out_dir, exist_ok=True) 
print(out_dir)

#%% load data

# load data
df_1 = pd.read_csv(os.path.join(prj_dir,'results','Run_8','metadata_te.csv'))#,index_col=0)
df_2 = pd.read_csv(os.path.join(prj_dir,'results','Run_8_inv','metadata_te.csv'))

if dataset == 'mindset':
    df_efa = pd.read_csv(os.path.join(datadir,'mindset_scores_all_20250415.csv'))
    cols = ['ML1', 'ML2', 'ML3', 'ML4']
elif dataset == 'mindset_nocogn':
    df_efa = pd.read_csv(os.path.join(datadir, 'mindset_scores_all_nocogn_20250417.csv'))
    cols = ['ML1', 'ML2', 'ML3']
elif dataset == 'become':
    df_efa = pd.read_csv(os.path.join(datadir,'become_efa_scores_all_20250520.csv'))
    cols = ['FAC1_1', 'FAC2', 'FAC3']
elif dataset == 'ima_str':
    df_efa = pd.read_csv(os.path.join(datadir,'imagen_scores_all_20250415.csv'))
    cols = ['ML1', 'ML2', 'ML3']
else:
    print("I don't know what to do")
    sys.exit()


X2_1 = ptk.dataio.fileio.load(os.path.join(Zdir, 'Z_ref.nii.gz'), mask=mask_nii, vol=False).T
X2_2 = ptk.dataio.fileio.load(os.path.join(Zdir_2, 'Z_ref.nii.gz'), mask=mask_nii, vol=False).T



#%% Combine EFA data from both runs

# Exclude subjects with missing factor scores
df_efa = df_efa.dropna()

# Convert subj_id column
df_efa['subj_id'] = df_efa['subj_id'].astype('str')

# Add index columns to the metadata dfs
df_1 = df_1.reset_index().rename(columns={'index': 'df1_index'})
df_2 = df_2.reset_index().rename(columns={'index': 'df2_index'})

# Merge EFA df with the metdata df (only keeps subjects present in EFA dataframe)
df_efa_1 = df_efa.merge(df_1[['subj_id', 'df1_index']], on='subj_id', how='inner')
df_efa_2 = df_efa.merge(df_2[['subj_id', 'df2_index']], on='subj_id', how='inner')


# Sort by indices from df_1 and df_2 (subjects must be in the same order as in the metadata and Z score files)
df_efa_1 = df_efa_1.sort_values(by='df1_index').reset_index(drop=True)
df_efa_2 = df_efa_2.sort_values(by='df2_index').reset_index(drop=True)

# convert to numpy array
X1_1 = df_efa_1[cols].to_numpy()
X1_2 = df_efa_2[cols].to_numpy()


# concatenate EFA data from both runs
if rep is None:
    print('combining both replications')
    X1 = np.concatenate((X1_1, X1_2), axis=0)
elif rep == 1:
    print('replication 1')
    X1 = X1_1
elif rep == 2:
    print('replication 2')
    X1 = X1_2
else:
    print("I don't know what to do")

##FACTORS
# # imagen
# #X1 = df_efa[['ML1', 'ML2', 'ML3']].to_numpy()
# # become 
# X1 = df_efa[['FAC1_1', 'FAC2_1', 'FAC3_1', 'FAC4_1', 'FAC5_1', 'FAC6_1']].to_numpy()
# # mindset
# #X1 = df_efa[['ML1', 'ML2', 'ML3', 'ML4']].to_numpy()


# select subjects from X2
idx_1 = df_efa_1['df1_index'].to_numpy()
X2_1 = X2_1[idx_1,:]


idx_2 = df_efa_2['df2_index'].to_numpy()
X2_2 = X2_2[idx_2,:]

if rep is None:
    X2 = np.concatenate((X2_1, X2_2), axis=0)
elif rep == 1:
    X2 = X2_1
elif rep == 2:
    X2 = X2_2
else:
    print("I don't know what to do")

#%% check for nan
np.isnan(X1).any()
np.isnan(X2).any()

(X1 == 0).any()
(X2 == 0).any()

_, col_indices = np.where(X2 == 0)
np.unique(col_indices)

#%% Impute zeros in X2

# impute Z
X2_imputed = np.copy(X2)

for col in range(X2.shape[1]):
    # Get the column values
    col_values = X2[:, col]
        
    # Find indices of columns with null values
    null_indices = np.where(col_values == 0)[0]
        
    # Generate random normal values 
    random_values = np.random.normal(loc=0, scale=1, size=len(null_indices))
        
    # Replace nulls with random values
    X2_imputed[null_indices, col] = random_values

X2 = X2_imputed

np.save(os.path.join(out_dir,'X1.npy'),X1)
np.save(os.path.join(out_dir,'X2.npy'),X2)

#%% Run scca 

# SSCCA config
l1 = [0.99, 0.5]
sign = [0., 0.]
n_comp = 1
rank = 3
n_splits = 25

W1 = np.zeros((n_splits, X1.shape[1], rank))
W2 = np.zeros((n_splits, X2.shape[1], rank))
R2_Test = np.zeros((n_splits,X1.shape[1], rank))
R2_Train = np.zeros((n_splits,X1.shape[1], rank))
R2v_Test = np.zeros((n_splits,X2.shape[1], rank))
R2v_Train = np.zeros((n_splits,X2.shape[1], rank))
R = np.zeros((n_splits, rank))
R_Train = np.zeros((n_splits, rank))
for i in range(n_splits):
    tr = np.random.uniform(size=X1.shape[0]) < 0.7
    te = ~tr
    
    # standardize x1
    if sign[0] == 1:
        X1tr = X1[tr,:]
        X1te = X1[te,:]
    else:
        m1 = np.mean(X1[tr,:], axis = 0)
        s1 = np.std(X1[tr,:], axis = 0)
        X1tr = (X1[tr,:] - m1) / s1
        X1te = (X1[te,:] - m1) / s1
    
    # standardize x2
    m2 = np.mean(X2[tr,:], axis = 0)
    s2 = np.std(X2[tr,:], axis = 0)
    X2tr = (X2[tr,:] - m2) / s2
    X2te = (X2[te,:] - m2) / s2

    print('split', i, 'fitting scca...')
    Cm = MSCCA(n_components=rank, n_views=2)
    Cm.fit([X1tr, X2tr], l1=l1, sign=sign, verbose=False, rank=rank)
    
    scores_te = Cm.transform([X1te, X2te])

    # compute cross-loadings
    print('computing cross-loadings')
    for rr in range(rank):
        for j in range(X1tr.shape[1]):
            R2_Train[i,j,rr]= stats.pearsonr(Cm.scores[1][:,rr], X1tr[:,j]).statistic ** 2
            R2_Test[i,j,rr]= stats.pearsonr(scores_te[1][:,rr], X1te[:,j]).statistic ** 2
        # compute voxe-wise loadings
        R2v_Train[i,:,rr]= fast_r(np.tile(Cm.scores[0][:,rr], (X2tr.shape[1],1)).T, X2tr) ** 2
        R2v_Test[i,:,rr]= fast_r(np.tile(scores_te[0][:,rr],(X2te.shape[1],1)).T, X2te) ** 2
    
    r_test = np.zeros(rank)
    r_train = np.zeros(rank)
    for rr in range(rank):
        r_test[rr] = stats.pearsonr(scores_te[0][:,rr],scores_te[1][:,rr]).statistic
        r_train[rr] = stats.pearsonr(Cm.scores[0][:,rr],Cm.scores[1][:,rr]).statistic
    
    
    print('r_train =', r_train, 'r_test =',r_test)
    W1[i,:,:] = Cm.W[0]
    W2[i,:,:] = Cm.W[1]
    R[i,:] = r_test
    R_Train[i,:] = r_train

print('canonical correlation:', np.mean(R,axis=0))

# # Find indices where both x1_scores and x2_scores are not NaN
# valid_indices = ~np.isnan(x1_scores) & ~np.isnan(x2_scores)

# # Calculate correlation only on valid indices (where both scores are not NaN)
# r = np.corrcoef(x1_scores[valid_indices], x2_scores[valid_indices])[0, 1]

#%% SAVE OUTPUT

np.save(os.path.join(out_dir, 'W_symptoms.npy'), W1)
np.save(os.path.join(out_dir, 'W_brain.npy'), W2)
np.save(os.path.join(out_dir, 'R2_symptoms_test.npy'), R2_Test)
np.save(os.path.join(out_dir, 'R2_symptoms_train.npy'), R2_Train)

for i in range(rank):
    # weights
    ptk.dataio.fileio.save(W2[:,:,i], os.path.join(out_dir,'W_rank' + str(i)+ '.nii.gz'), example=ex_nii, mask=mask_nii)
    stable_features = np.sum(abs(W2)>0,axis=0)/n_splits >= 0.8

    W2s = W2
    W2s[:,~stable_features] = 0
    ptk.dataio.fileio.save(W2s[:,:,i], os.path.join(out_dir,'Ws_rank' + str(i) + '.nii.gz'), example=ex_nii, mask=mask_nii)

    # structure coefficients from brain (train)
    ptk.dataio.fileio.save(R2v_Train[:,:,i], os.path.join(out_dir,'R2_brain_train_rank' + str(i) + '.nii.gz'), example=ex_nii, mask=mask_nii)
    R2s = R2v_Train
    R2s[:,~stable_features] = 0
    ptk.dataio.fileio.save(R2s[:,:,i], os.path.join(out_dir,'R2s_brain_train_rank' + str(i) + '.nii.gz'), example=ex_nii, mask=mask_nii)
    
    # structure coefficients from brain (test)
    ptk.dataio.fileio.save(R2v_Test[:,:,i], os.path.join(out_dir,'R2_brain_test_rank' + str(i) + '.nii.gz'), example=ex_nii, mask=mask_nii)
    R2s = R2v_Test
    R2s[:,~stable_features] = 0
    ptk.dataio.fileio.save(R2s[:,:,i], os.path.join(out_dir,'R2s_brain_test_rank' + str(i) + '.nii.gz'), example=ex_nii, mask=mask_nii)


r_test = pd.DataFrame(R)
r_train = pd.DataFrame(R_Train)
r_test.to_csv(os.path.join(out_dir, 'r_test.csv'),index=False)
r_train.to_csv(os.path.join(out_dir, 'r_train.csv'), index = False)

