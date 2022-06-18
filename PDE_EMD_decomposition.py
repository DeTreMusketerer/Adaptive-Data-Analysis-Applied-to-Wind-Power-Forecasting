# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:43:51 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, the wind power data is decomposed with the PDE-EMD method
using multiprocessing.
"""


import numpy as np
import multiprocessing as mp
import time as time
import Modules.PDE_EMD as PDE


def get_result(result):
    """
    

    Parameters
    ----------
    result : tuple
        tuple containing the current index and the decomposition of the window
        with that index.

    Returns
    -------
    None.

    """
    
    global IMFs
    IMFs[result[0],  :, :] = result[1]
    if result[0] % 100 == 0:
        np.save("Data/PDE_Window_temp.npy", IMFs[:result[0], :, :])


def live_PDE_EMD(t, spatialline, signal, q=288, T=6, max_imf=9, boundary = "Neumann_0"):
    """
    Parameters
    ----------
    t : int
        index of the window which is being decomposed.
    spatialline : ndarray
        array of spatial indicies.
    signal : ndarray
        signal which should be decomposed.
    q : int, optional
        windowlength. The default is 288.
    T : float, optional
        End time. The default is 6.
    max_imf : int, optional
        maximum amount of IMFs which should be found. The default is 9.
    boundary : str, optional
        Choses which type of boundary condition is used. Currently, Dirichlet_0,
        Dirichlet_1, Neumann_0 and Neumann_1 are implemented.
        The default is "Neumann_0".

    Returns
    -------
    t : int
        index of the window which is being decomposed..
    PDE_window : ndarray
        decomposition of the t'th window.

    """
    
    assert (len(signal) >= q), "The length of the signal must be longer than the size of the window!"
    if t + q >= len(signal):
        assert False, "Not enough previous data for this choice of window length!"
    else:
        PDE_window = PDE.PDE_EMD().PDE_EMD(spatialline[t:t+q], signal[t:t+q],
                                           T, max_IMF = max_imf, boundary = boundary)
        tmp = np.shape(PDE_window)[0]

        if tmp < max_imf+1:
            zeros = np.zeros((max_imf+1-tmp,q), dtype=np.float32)
            PDE_window = np.vstack((PDE_window, zeros))
        
        assert np.shape(PDE_window)[0] > 1, \
                    "Unable to decompose signal. Try increasing the window length!"
        return (t, PDE_window)


if __name__ == "__main__":
    days = 2
    q = 288*days
    max_imf = 10
    data_type = "train"
    if data_type == "train":
        y = np.load("Data/training_data.npy")
        mesh = np.load(f"Data/train_mesh_q{q}.npy")
    else:
        y = np.load("Data/test_data.npy")
        mesh = np.load(f"Data/test_mesh_q{q}.npy")
    
    n = len(y)
    IMFs = np.zeros((n, max_imf+1, q), dtype=np.float32)
    spatialline = np.linspace(0,days,n)

    M = mp.cpu_count()
    max_cpu = 8 # determines the maximum amount of CPUs that will be used.
    if M > max_cpu:
        M = max_cpu

    #Hyperparameters
    T = 6
    boundary = "Neumann_0"

    t1 = time.time()
    pool = mp.Pool(M)

    for t in range(n):
        if mesh[t] == 1:
            res = pool.apply_async(live_PDE_EMD, args=(t, spatialline, y, q, T,
                                                       max_imf, boundary),
                                    callback=get_result)
            if t % (50*M) == 0 and t != 0: # We still want to keep somewhat track of progress
                print(f"{t}")
                res.wait()
    pool.close()
    pool.join()
    print(f"Processing time {time.time()-t1} with {M} CPUs.")
    if data_type == "train":
        np.save(f"Data/PDE_Window_q{q}_T{T}_{boundary}.npy", IMFs)
    else:
        np.save(f"Data/PDE_Window_Test_q{q}_T{T}_{boundary}.npy", IMFs)

