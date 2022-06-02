"""
In this script the 95th percentile for the used methods is calculated
"""


import numpy as np


methods = ["error_EMD-LSTM128", "test_error_FFT-NMP-EMD-LSTM002", "test_error_EMD-Live-LSTM057",
            "test_error_FFT-NMP-EMD-Live-LSTM002", "test_error_PDE-EMD-Live-LSTM089"]
percentile = 0.95
CAP = np.load("Data/DK1-1_Capacity.npy")/1000

model = methods[1]
epsilon_test = np.load(f"results/{model}.npy")
epsilon_test_entry = np.sum(epsilon_test, axis = 1)/CAP
n_test = len(epsilon_test_entry)
eps_sort = np.sort(abs(epsilon_test_entry))
sample = int(n_test*percentile)
zeta = eps_sort[sample]

## LSTM
epsilon_LSTM = np.load("results/test_error_LSTM032.npy")/CAP
eps_sort_LSTM = np.sort(abs(epsilon_LSTM))
n_test = len(eps_sort_LSTM)
sample = int(n_test*percentile)
zeta_LSTM = eps_sort_LSTM[sample]

## AR
epsilon_AR = np.load("results/AR_error.npy")
eps_12 = epsilon_AR[11,:,0]/CAP
eps_sort_AR = np.sort(abs(eps_12))
n_test = len(eps_sort_AR)
sample = int(n_test*percentile)
zeta_AR = eps_sort_AR[sample]
