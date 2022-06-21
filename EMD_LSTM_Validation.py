# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:42:06 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, early stopping is done for EMD-LSTM models using the subtraining
data and the validation data in an online setup.
The script can also be used for the NMP-EMD-LSTM and FFT-NMP-EMD-LSTM models
by changing the data imported.

Track changes:
    version 1.0: Basic adaptation of the online training script. This is
                 still not completely ready for use. (20/04/2022)
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Modules.NN_module import LSTM, GRU, PyTorchDataset_RealTime, early_stopping, test


if __name__ == '__main__':
    # =============================================================================
    # Parameter specifications
    # =============================================================================
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_epochs = 100 # Maximum number of epochs.
    input_size = 3 # Number of samples in a training dataset.
    tau = 12 # Number of samples we predict ahead.
    batch_size = 32
    learning_rate = 1e-04 # Initial learning rate.
    hidden_sizes = [64, 64, 32] # Number of hidden units in hidden layers.
    dropout_hidden = 0.3 # Dropout rate.
    gamma = 0.9 # Learning rate decay.
    log_interval = 600 # Interval for logging validation loss during early stopping.
    patience = 60 # Patience parameter for early stopping.
    q = 288 # Window length
    xi = 0 # Shift of window for target
    Type = "LSTM" # Neural network model type, options "LSTM", "GRU".
    model_name = "FFT-NMP-EMD-Live-LSTM001"

    # =============================================================================
    # Import data
    # =============================================================================
    #y_train = np.load(f"Data/EMD_Window_q{q}_training_data.npy")
    y_train = np.load("Data/IMFs_FFT-NMP-EMD001_training.npy")
    train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
    power_sub = np.load("Data/subtraining_data.npy")
    n_sub = np.shape(power_sub)[0]

    y_sub = y_train[:n_sub, :, :]
    sub_mesh = train_mesh[:n_sub]

    y_val = y_train[n_sub:, :, :]
    val_mesh = train_mesh[n_sub:]

    # =============================================================================
    # Save model settings
    # =============================================================================
    with open(f"results/{model_name}.txt", "w") as file:
        file.write(f"Window length: {q}\n")
        file.write(f"Input size: {input_size}\n")
        file.write(f"Tau: {tau}\n")
        file.write(f"xi: {xi}\n")
        file.write("Hidden sizes: ")
        for h in hidden_sizes:
            file.write(f"{h}, ")
        file.write("\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Initial learning rate: {learning_rate}\n")
        file.write(f"Learning rate decay: {gamma}\n")
        file.write(f"Dropout rate: {dropout_hidden}\n")
        file.write(f"Logging validation loss interval: {log_interval}\n")
        file.write(f"Early stopping patience: {patience}\n")
        file.write(f"Max epochs: {max_epochs}\n")
        file.write(f"Random seed: {seed}\n")

    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    _, s, _ = np.shape(y_train)
    dset3 = PyTorchDataset_RealTime(y_val[:, 0, :], val_mesh, xi, input_size, tau)
    valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
    n_val = len(valid_loader.dataset)
    epsilon = np.zeros((n_val, s), dtype=np.float32)

    for k in range(s):
        dset2 = PyTorchDataset_RealTime(y_sub[:, k, :], sub_mesh, xi, input_size, tau)
        subtrain_loader = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
        dset3 = PyTorchDataset_RealTime(y_val[:, k, :], val_mesh, xi, input_size, tau)
        valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)

        print(f"Beginning component {k}")
        model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)
    
        # Training
        optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
    
        opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
            model, device, optimiser, scheduler, subtrain_loader, valid_loader,
            log_interval, patience, max_epochs, hidden_sizes)

        print("Minimum validation loss: ", min_valid_loss)

        with open(f"results/{model_name}.txt", "a") as file:
            file.write(f"\nComponent {k}\n")
            file.write(f"Optimum number of parameter update (opt_upd): {opt_upd}\n")
            file.write(f"Subtrain updates pr epoch (upd_epoch): {upd_epoch}\n")
            file.write(f"Minimum validation loss: {min_valid_loss}\n")
            file.write(f"Component energy: {np.mean(y_val[:, k, :]**2)}\n")

        # Save learning curves
        with open(f'results/validation_loss_{model_name}_{k}.npy', 'wb') as file:
            np.save(file, valid_loss)
        with open(f'results/training_loss_{model_name}_{k}.npy', 'wb') as file:
            np.save(file, training_loss)

        epsilon[:, k] = test(model, device, valid_loader, hidden_sizes)

    tot_val_loss = np.mean(np.sum(epsilon, axis=1)**2)
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nTotal validation loss: {tot_val_loss}")