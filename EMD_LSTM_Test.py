# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:16:52 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, EMD-LSTM models are trained on the
training data and evaluated on the test data in an online setup.
The script can also be used for the NMP-EMD-LSTM and FFT-NMP-EMD-LSTM models
by changing the data imported.

Track changes:
    version 1.0: Basic implementation of neural network training and test script
                 for decomposition based models in realtime setting. (29/04/2022)
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Modules.NN_module import LSTM, GRU, PyTorchDataset_RealTime, test, early_stopping_retrain, test_pred_zero


if __name__ == '__main__':
    model_name = "EMD-Live-LSTM001"

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
        _ = int(line.split()[-1])
        line = file.readline()
        _ = int(line.split()[-1])
        line = file.readline()
        seed = int(line.split()[-1])

    load_model = False
    save_model = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    # =============================================================================
    # Import data
    # =============================================================================
    y_train = np.load(f"Data/EMD_Window_q{q}_training_data.npy")
    y_test = np.load(f"Data/EMD_Window_q{q}_test_data.npy")

    train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
    test_mesh = np.load(f"Data/test_mesh_q{q}.npy")

    CAP = np.load("Data/DK1-1_Capacity.npy")/1000 #kW to MW

    # =============================================================================
    # Read result from pre-training
    # =============================================================================
    _, s, _ = np.shape(y_train)
    opt_upd = np.zeros(s)
    upd_epoch = np.zeros(s)
    pred_zeros = np.zeros(s)
    with open(f"results/{model_name}.txt", "r") as file:
        line = ""
        while "Random seed" not in line:
            line = file.readline()
        seed = int(line.split()[-1])
        for k in range(s):
            while f"Component {k}" not in line:
                line = file.readline()
            line = file.readline()
            if "Predict zeros" in line:
                pred_zeros[k] = 1
            else:
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
    epsilon = np.zeros((n_test, s), dtype=np.float32)

    with open(f"results/{model_name}.txt", "a") as file:
        file.write("\nTesting\n")

    for k in range(s):
        dset1 = PyTorchDataset_RealTime(y_train[:, k, :], train_mesh, xi, input_size, tau)
        train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
        dset4 = PyTorchDataset_RealTime(y_test[:, k, :], test_mesh, xi, input_size, tau)
        test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)

        if pred_zeros[k] == 0:
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
    NRMSE = np.sqrt(np.mean(np.sum(epsilon, axis=1)**2/CAP**2))
    NMAE = np.mean(np.abs(np.sum(epsilon, axis=1))/CAP)
    NBIAS = np.mean(np.sum(epsilon, axis=1)/CAP)
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nTest MSE: {test_MSE}\n")
        file.write(f"NRMSE: {NRMSE}\t NMAE: {NMAE}\t NBIAS: {NBIAS}")

    np.save(f"results/test_error_{model_name}.npy", epsilon)

