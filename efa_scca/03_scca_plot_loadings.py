#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:13:12 2025

@author: andmar
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

wdir = '/project/3022054.01/projects/linschl/results/CCA/Run_5/become_v2_nopos/'

R2_Train = np.load(os.path.join(wdir,'R2_symptoms_train.npy'))
R2_Test = np.load(os.path.join(wdir,'R2_symptoms_test.npy'))
r_train = np.loadtxt(os.path.join(wdir,'r_train.csv'),delimiter=',', skiprows=1)
r_test =  np.loadtxt(os.path.join(wdir,'r_test.csv'),delimiter=',', skiprows=1)

n_splits, d, rank = R2_Train.shape
s = 0
for s in range(n_splits):
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    for r in range(R2_Train.shape[2]):
        if r == 0: 
            ax1.bar(np.arange(d), R2_Train[s,:,r])
            ssum = R2_Train[s,:,r]
        else:
            ax1.bar(np.arange(d), R2_Train[s,:,r] + ssum,
                    bottom = ssum)
            ssum = ssum + R2_Train[s,:,r]
        ax1.set_title(f'Train {np.array2string(r_train[s,:], precision=2)}')
        plt.legend(('CV1', 'CV2', 'CV3'))
    for r in range(R2_Train.shape[2]):
        if r == 0: 
            ax2.bar(np.arange(d), R2_Test[s,:,r])
            ssum = R2_Test[s,:,r]
        else:
            ax2.bar(np.arange(d), R2_Test[s,:,r] + ssum,
                    bottom = ssum)
            ssum = ssum + R2_Test[s,:,r]
        ax2.set_title(f'Test {np.array2string(r_test[s,:], precision=2)}')
        plt.legend(('CV1', 'CV2', 'CV3'))
    plt.show()

totalR2 = np.sum(R2_Test,axis=2)
plt.bar(np.arange(d), np.mean(totalR2, axis=0))
plt.errorbar(np.arange(d), np.mean(totalR2, axis=0), yerr=np.std(totalR2, axis=0), fmt=".", color="k")
plt.show()
print('train:', np.mean(r_train,axis=0), 'test:', np.mean(r_test,axis=0))


#%% save R2 for figures

totalR2 = np.sum(R2_Test, axis=2)
R2_df = pd.DataFrame(totalR2)
R2_df.to_csv(os.path.join(wdir,'R2_total_test.csv'), index = False)
