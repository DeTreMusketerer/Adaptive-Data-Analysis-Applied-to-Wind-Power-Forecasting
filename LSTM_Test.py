# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:46:31 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, LSTM models are trained on the
training data and evaluated on the test data.

Track changes:
    version 1.0: Basic implementation of neural network training and test script
                 for baseline LSTM model. (29/04/2022)
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Modules.NN_module import LSTM, GRU, PyTorchDataset, test, early_stopping_retrain


if __name__ == '__main__':
    model_name = "LSTM001"

    # =============================================================================
    # Read model settings
    # =============================================================================
    with open(f"results/{model_name}.txt", "r") as file:
        line = file.readline()
        input_size = int(line.split()[-1])
        line = file.readline()
        tau = int(line.split()[-1])
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
        line = file.readline()
        opt_upd = int(line.split()[-1])
        line = file.readline()
        upd_epoch = int(line.split()[-1])

    load_model = False
    save_model = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    # =============================================================================
    # Import data
    # =============================================================================
    y_train = np.load("Data/training_data.npy")
    train_mesh = np.load("Data/train_mesh_not_realtime.npy")
    y_max = np.amax(y_train)

    y_test = np.load("Data/test_data.npy")
    test_mesh = np.load("Data/test_mesh_not_realtime.npy")

    dset1 = PyTorchDataset(y_train, train_mesh, input_size, tau)
    train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
    dset4 = PyTorchDataset(y_test, test_mesh, input_size, tau)
    test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)

    # =============================================================================
    # Training
    # =============================================================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)
    if load_model is True:
        model.load_state_dict(torch.load('models/' + model_name + '.pt', map_location=device))

    print('\n Re-training:\n')
    optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
    scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

    updates_counter = 0
    epoch = 1
    while updates_counter < opt_upd:
        updates_counter = early_stopping_retrain(model, device, train_loader,
                                                 optimiser, epoch, opt_upd,
                                                 updates_counter, scheduler,
                                                 upd_epoch, log_interval, hidden_sizes)
        print('')
        epoch += 1

    # Save model?
    if save_model is True:
        torch.save(model.state_dict(), 'models/' + model_name + '.pt')

    # =============================================================================
    # Testing
    # =============================================================================
    print('Testing')
    epsilon = test(model, device, test_loader, hidden_sizes)

    CAP = np.load("Data/DK1-1_Capacity.npy")/1000 #kW to MW
    test_MSE = np.mean(epsilon**2)
    NRMSE = np.sqrt(np.mean(epsilon**2/CAP**2))
    NMAE = np.mean(np.abs(epsilon)/CAP)
    NBIAS = np.mean(epsilon/CAP)
    print(f"MSE: {test_MSE}\tNRMSE: {NRMSE}\tNMAE: {NMAE}\tNBIAS: {NBIAS}")
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nMSE: {test_MSE}\nNRMSE: {NRMSE}\nNMAE: {NMAE}\nNBIAS: {NBIAS}\n")
    np.save(f"results/test_error_{model_name}.npy", epsilon)

