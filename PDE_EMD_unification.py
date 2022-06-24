"""
Created on Thur Apr 21 13:12:06 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, the unification procedure for the PDE-EMD method is implemented
as described in the report
        Adaptive Data Analysis:
        Theoretical Results and an Application to Wind Power Forecasting
            - Appendix C: Unification Procedure
"""

import numpy as np
import Modules.PDE_EMD as PDE

q = 576
T = 6
s = 5
boundary = "Neumann_0"
data_type = "test"
if data_type == "train":
    decomposition = np.load(f"Data/PDE_Window_q{q}_T{T}_{boundary}.npy")
else:
    decomposition = np.load(f"Data/PDE_Window_Test_q{q}_T{T}_{boundary}.npy")
n, m, q = np.shape(decomposition)
train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
test_mesh = np.load(f"Data/test_mesh_q{q}.npy")
   

# Unification in residual
T_line = np.arange(0, n)
unified = np.zeros((n,s+1,q)).astype(np.float32)
if data_type == "train":
    try:
        unified = np.load(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified[i,s,:] = np.sum(decomposition[i,s:,:], axis = 0)
            for j in range(s):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified[i,s,:] += decomposition[i,j,:]
                else:
                    unified[i,j,:] = decomposition[i,j,:]
        np.save(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy", unified)
else:
     try:
         unified = np.load(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
     except  FileNotFoundError:
         for i in range(n):
             unified[i,s,:] = np.sum(decomposition[i,s:,:], axis = 0)
             for j in range(s):
                 max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                 if len(max_pos) + len(min_pos) < 4:
                     unified[i,s,:] += decomposition[i,j,:]
                 else:
                     unified[i,j,:] = decomposition[i,j,:]
         np.save(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy", unified)