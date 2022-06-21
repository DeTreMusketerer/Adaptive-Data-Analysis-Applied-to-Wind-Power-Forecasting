# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:32:23 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Decomposition based univariate-RNN on the danish wind power production with
the decomposition being made offline (on the entire dataset).
"""
import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from NN_module import LSTM, PyTorchDataset, test, early_stopping_retrain, test_persistence


if __name__ == '__main__':
    #model_name = "FFT-NMP-EMD-LSTM002"

    # =============================================================================
    # Read model settings
    # =============================================================================

    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # input_size = 12 # Number of samples in a training dataset.
    # tau = 12 # Number of samples we predict ahead.
    # batch_size = 32
    # learning_rate = 1e-04 # Initial learning rate.
    # hidden_sizes = [512, 256, 128] # Number of hidden units in hidden layers.
    # dropout_hidden = 0.1 # Dropout rate.
    # gamma = 0.7 # Learning rate decay.
    # log_interval = 100 # Interval for logging validation loss during early stopping.
    # patience = 100 # Patience parameter for early stopping.

    Type = "LSTM" # Neural network model type, options "LSTM", "GRU".
    model_name = "EMD-LSTM001"
    
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
        patience = int(line.split()[-1])
        line = file.readline()
        epochs = int(line.split()[-1])
        line = file.readline()
        seed = int(line.split()[-1])


    load_model = False
    save_model = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    # =============================================================================
    # Import data
    # =============================================================================
    y_train = np.load("Data/EMD_full_training_data.npy")
    train_mesh = np.load("Data/train_mesh_not_realtime.npy")

    train_mesh = np.load("Data/train_mesh_not_realtime.npy")
    test_mesh = np.load("Data/test_mesh_not_realtime.npy")

    y_test = np.load("Data/EMD_full_test_data.npy")


    CAP = np.load("Data/DK1-1_Capacity.npy")/1000 #kW to MW

    # =============================================================================
    # Read result from pre-training
    # =============================================================================
    _, s = np.shape(y_train)
    opt_upd = np.zeros(s)
    upd_epoch = np.zeros(s)
    with open(f"results/{model_name}.txt", "r") as file:
        line = ""
        for k in range(s-1):
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

    _, s = np.shape(y_train)
    dset3 = PyTorchDataset(y_test[:, 0], test_mesh, input_size, tau)
    valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
    n_test = len(valid_loader.dataset)
    epsilon = np.zeros((n_test, s), dtype=np.float32)

    with open(f"results/{model_name}.txt", "a") as file:
        file.write("\nTesting\n")

    for k in range(s):
        if k < s-1:
            dset1 = PyTorchDataset(y_train[:, k], train_mesh, input_size, tau)
            train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
            dset4 = PyTorchDataset(y_test[:, k], test_mesh, input_size, tau)
            test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
    
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
        elif k == s-1:
            dset4 = PyTorchDataset(y_test[:, k], test_mesh, input_size, tau)
            test_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
            epsilon[:, k] = test_persistence(device, test_loader)

        with open(f"results/{model_name}.txt", "a") as file:
            file.write(f"Component {k} test MSE: {np.mean(epsilon[:, k]**2)}\n")


    np.save(f"results/error_{model_name}.npy", epsilon)

    test_MSE = np.mean(np.sum(epsilon, axis=1)**2)
    NRMSE = np.sqrt(np.mean(np.sum(epsilon, axis=1)**2/CAP**2))
    NMAE = np.mean(np.abs(np.sum(epsilon, axis=1))/CAP)
    NBIAS = np.mean(np.sum(epsilon, axis=1)/CAP)
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nTest MSE: {test_MSE}\n")
        file.write(f"NRMSE: {NRMSE}\t NMAE: {NMAE}\t NBIAS: {NBIAS}")

    np.save(f"results/test_error_{model_name}.npy", epsilon)


