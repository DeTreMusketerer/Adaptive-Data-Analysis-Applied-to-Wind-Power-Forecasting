"""
In this script unification procedure is done for the PDE-EMD
"""

import numpy as np
import Modules.PDE_EMD as PDE

q = 288
T = 6
s = 4
boundary = "Neumann_0"
data_type = "train"
if data_type == "train":
    decomposition = np.load(f"Data/PDE_Window_q{q}_T{T}_{boundary}.npy")
else:
    decomposition = np.load(f"Data/PDE_Window_Test_q{q}_T{T}_{boundary}.npy")
n, m, q = np.shape(decomposition)
train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
test_mesh = np.load(f"Data/test_mesh_q{q}.npy")


# Unification in IMFs
T_line = np.arange(0, n)
unified_4 = np.zeros((n,4,q)).astype(np.float32)
if data_type == "train":
    try:
        unified_4 = np.load(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified_4[i,3,:] = np.sum(decomposition[i,4:,:], axis = 0)
            for j in range(4):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified_4[i,3,:] += decomposition[i,j,:]
                elif j == 2 or j == 3:
                    unified_4[i,2,:] += decomposition[i,j,:]
                else:
                    unified_4[i,j,:] += decomposition[i,j,:]
        np.save(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy", unified_4)
else:
    try:
        unified_4 = np.load(f"Data/PDE_Window_IMFs_Test_q{q}_T{T}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified_4[i,3,:] = np.sum(decomposition[i,4:,:], axis = 0)
            for j in range(s):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified_4[i,3,:] += decomposition[i,j,:]
                elif j == 2 or j == 3:
                    unified_4[i,2,:] += decomposition[i,j,:]
                else:
                    unified_4[i,j,:] += decomposition[i,j,:]
        np.save(f"Data/PDE_Window_IMFs_Test_q{q}_T{T}_{boundary}.npy", unified_4)    

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