# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:24:01 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, the wind power data is decomposed with the EMD method
using multiprocessing.
"""


import numpy as np
import Modules.HilbertHuangTransform as HHT


if __name__ == "__main__":
    days = 1
    q = 288*days
    max_imf = 10
    data_type = "training"
    if data_type == "training":
        y = np.load("Data/training_data.npy")
        mesh = np.load(f"Data/train_mesh_q{q}.npy")
    else:
        y = np.load("Data/test_data.npy")
        mesh = np.load(f"Data/test_mesh_q{q}.npy")

    IMFs_init = HHT.decompose_EMD_MP(y, mesh, model_name=f"{data_type}_q{q}", q=q, max_imf=9)
    print("Decomposition Done")

    s = 6
    IMFs_fixed = HHT.unification_procedure(IMFs_init, s)
    print("Unification Done")

    np.save(f"Data/EMD_Window_q{q}_{data_type}_data.npy", IMFs_fixed)
