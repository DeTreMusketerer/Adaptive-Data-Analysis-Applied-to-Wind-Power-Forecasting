# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:43:12 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, the wind power data is decomposed with the FFT-NMP-EMD and
NMP-EMD methods using multiprocessing.
"""


import numpy as np
from Modules.NonlinearMatchingPursuit import decompose_NMP_MP


if __name__ == "__main__":
    model_name = "FFT-NMP-EMD001"
    lambda_K = 0.4
    use_fast_alg = True
    q = 288
    s = 4
    y_train = np.load("Data/training_data.npy")
    train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
    decompose_NMP_MP(y_train, train_mesh, model_name, s, q, use_fast_alg, lambda_K)

