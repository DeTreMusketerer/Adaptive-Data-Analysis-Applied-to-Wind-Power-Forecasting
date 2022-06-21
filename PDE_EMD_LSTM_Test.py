# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:08:57 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, PDE-EMD-LSTM models are trained on the
training data and evaluated on the test data in an online setup.
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Modules.NN_module import LSTM, PyTorchDataset_RealTime, test, early_stopping_retrain, test_pred_zero


if __name__ == '__main__':
    model_name = "PDE-EMD-Live-LSTM089"
    # =============================================================================
    # Read model settings
    # =============================================================================
    with open(f"results/{model_name}.txt", "r") as file:
        line = file.readline()
        q = int(line.split()[-1])
        line = file.readline()
        input_size = int(line.split()[-1])
        line = file.readline()
        tau = int(line.split()[-1])
        line = file.readline()
        xi = int(line.split()[-1])
        line = file.readline()
        hidden_sizes = [eval(b) for b in line.split(":")[-1].split(",")[:-1]]
        line = file.readline()
        batch_size = int(line.split()[-1])
        line = file.readline()
        learning_rate = float(line.split()[-1])
        line = file.readline()
        gamma = float(line.split()[-1])
        line = file.readline()
        dropout_hidden = float(line.split()[-1])
        line = file.readline()
        log_interval = int(line.split()[-1])
        line = file.readline()
        patience = int(line.split()[-1])
        line = file.readline()
        seed = int(line.split()[-1])
        line = file.readline()
        T = line.split()[-1]
        line = file.readline()
        s = line.split()[-1]
        line = file.readline()
        boundary = line.split()[-1]
        line = file.readline()
        type_data = line.split()[-1]

    load_model = False
    save_model = True

    
    # =============================================================================
    # Import data
    # =============================================================================
    if type_data == "normal": # Becomes Reduce in residual when s = 2
        y_train = np.load(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
        y_test = np.load(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
    elif type_data == "IMF-unified": # Reduces in IMFs
        y_train = np.load(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy")
        y_test = np.load(f"Data/PDE_Window_IMFs_Test_q{q}_T{T}_{boundary}.npy")
    else:
        print("type_data variable not supported")
    
    train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
    test_mesh = np.load(f"Data/test_mesh_q{q}.npy")
    CAP = np.load("Data/DK1-1_Capacity.npy")

    # =============================================================================
    # Read result from pre-training
    # =============================================================================
    s = eval(s)
    opt_upd = np.zeros(s+1)
    upd_epoch = np.zeros(s+1)
    with open(f"results/{model_name}.txt", "r") as file:
        line = ""
        while "Random seed" not in line:
            line = file.readline()
        seed = int(line.split()[-1])
        for k in range(s+1):
            while f"Component {k}" not in line:
                line = file.readline()
            line = file.readline()
            opt_upd[k] = int(line.split()[-1])
            line = file.readline()
            upd_epoch[k] = int(line.split()[-1])

    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    dset4 = PyTorchDataset_RealTime(y_test[:, k, :], test_mesh, xi, input_size, tau)
    test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
    n_test = len(test_loader.dataset)
    epsilon = np.zeros((n_test, s+1), dtype=np.float32)
    for k in range(s+1):
        dset4 = PyTorchDataset_RealTime(y_test[:, k, :], test_mesh, xi, input_size, tau)
        test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
        if upd_epoch[k] != 0:
            dset1 = PyTorchDataset_RealTime(y_train[:, k, :], train_mesh, xi, input_size, tau)
            train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
    
            # Training
            print('\n Re-training:\n')
            print(f"Beginning component {k}")
            model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)
            if load_model is True:
                model.load_state_dict(torch.load(f"models/{model_name}_{k}.pt", map_location=device))
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
    
            updates_counter = 0
            epoch = 1
            while updates_counter < opt_upd[k]:
                updates_counter = early_stopping_retrain(model, device, train_loader,
                                        optimiser, epoch, opt_upd[k], updates_counter,
                                        scheduler, upd_epoch[k], log_interval, hidden_sizes)
                print('')
                epoch += 1
    
            # Save model?
            if save_model is True:
                torch.save(model.state_dict(), f"models/{model_name}_{k}.pt")

            # =============================================================================
            # Testing
            # =============================================================================
            print('Testing')
            epsilon[:, k] = test(model, device, test_loader, hidden_sizes)
            with open(f"results/{model_name}.txt", "a") as file:
                file.write(f"Component {k} test MSE: {np.mean(epsilon[:, k]**2)}\n")
        else:
            epsilon[:, k] = test_pred_zero(device, test_loader)
            with open(f"results/{model_name}.txt", "a") as file:
                file.write(f"Component {k} test MSE: {np.mean(epsilon[:, k]**2)}\n")


    test_MSE = np.mean(np.sum(epsilon**2, axis=1))
    NRMSE = np.sqrt(np.mean(np.sum(epsilon**2, axis=1)/CAP**2))
    NMAE = np.mean(np.sum(np.abs(epsilon/CAP), axis = 1))
    NBIAS = np.mean(np.sum(epsilon, axis=1)/CAP)
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nTest MSE: {test_MSE}\n")
        file.write(f"NRMSE: {NRMSE}\t NMAE: {NMAE}\t NBIAS: {NBIAS}")

    print(test_MSE)
    np.save(f"results/test_error_{model_name}.npy", epsilon)
