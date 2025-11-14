#%% Config
import os
import glob
import subprocess
import numpy as np
import pandas as pd

wdir =  '/project/3022054.01/projects/linschl/results/CCA/Run_4/mindset_v1_nopos'
wdir2 = '/project/3022054.01/projects/linschl/results/CCA/Run_4/mindset_v2_nopos'


#true_r = pd.read_csv(os.path.join(wdir,'r_test.csv')).mean(axis=0)[0]
true_r = np.sum(pd.read_csv(os.path.join(wdir,'r_test.csv')).mean(axis=0))

print('split 1:')
print(true_r)

pnames = glob.glob(os.path.join(wdir,'perm','r_perm_*'))
if len(pnames) > 1000 :
    pnames = pnames[:1000]
    n_perms = 1000
else:
    n_perms = len(pnames)

p_r = np.zeros(n_perms)
for i, p in enumerate(pnames):
    p_r[i] = np.loadtxt(p)

pval = sum(p_r > true_r) / n_perms
print(f'r={true_r} ({n_perms} permutations) = {pval}')

print('split 2:')
#true_r = pd.read_csv(os.path.join(wdir,'r_test.csv')).mean(axis=0)[0]
true_r2 = np.sum(pd.read_csv(os.path.join(wdir2,'r_test.csv')).mean(axis=0))

pnames = glob.glob(os.path.join(wdir,'perm','r_perm_*'))
if len(pnames) > 1000 :
    pnames = pnames[:1000]
    n_perms = 1000
else:
    n_perms = len(pnames)

p_r2 = np.zeros(n_perms)
for i, p in enumerate(pnames):
    p_r2[i] = np.loadtxt(p)

pval = sum(p_r2 > true_r2) / n_perms
print(f'r={true_r2} ({n_perms} permutations) = {pval}')

print('aggregate:')

maxi = min(len(p_r2), len(p_r))
true_r_agg = (true_r + true_r2) / 2
p_r_agg = (p_r[:maxi] + p_r2[:maxi]) / 2

pval = sum(p_r_agg > true_r_agg) / maxi
print(f'r={true_r_agg} ({maxi} permutations) = {pval}')
