# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:40:52 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Decomposition based univariate-RNN on the danish wind power production with
the decomposition being made offline (on the entire dataset).
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from NN_module import LSTM, PyTorchDataset, test, early_stopping, test_persistence


if __name__ == '__main__':
    # =============================================================================
    # Parameter specifications
    # =============================================================================
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_epochs = 100 # Maximum number of epochs.
    input_size = 12 # Number of samples in a training dataset.
    tau = 12 # Number of samples we predict ahead.
    batch_size = 32
    learning_rate = 1e-04 # Initial learning rate.
    hidden_sizes = [512, 256, 128] # Number of hidden units in hidden layers.
    dropout_hidden = 0.1 # Dropout rate.
    gamma = 0.7 # Learning rate decay.
    log_interval = 100 # Interval for logging validation loss during early stopping.
    patience = 100 # Patience parameter for early stopping.
    Type = "LSTM" # Neural network model type, options "LSTM", "GRU".
    model_name = "EMD-LSTM001"


    # =============================================================================
    # Import data
    # =============================================================================
    y_train = np.load("Data/EMD_full_training_data.npy")
    train_mesh = np.load("Data/train_mesh_not_realtime.npy")

    train_mesh = np.load("Data/train_mesh_not_realtime.npy")
    test_mesh = np.load("Data/test_mesh_not_realtime.npy")

    y_test = np.load("Data/EMD_full_test_data.npy")
    

    sub_mesh = np.load("Data/subtraining_mesh_not_realtime.npy")
    val_mesh = np.load("Data/validation_mesh_not_realtime.npy")

    y_sub = np.load("Data/EMD_full_subtraining_data.npy")
    y_val = np.load("Data/EMD_full_validation_data.npy")

    dset2 = PyTorchDataset(y_sub, sub_mesh, input_size, tau)
    subtrain_loader = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
    dset3 = PyTorchDataset(y_val, val_mesh, input_size, tau)
    valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)

    # =============================================================================
    # Save model settings
    # =============================================================================
    with open(f"results/{model_name}.txt", "w") as file:
        file.write(f"Input size: {input_size}\n")
        file.write(f"Tau: {tau}\n")
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
        file.write(f"Maximum epochs: {max_epochs}\n")
        file.write(f"Random seed: {seed}\n")

    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    _, s = np.shape(y_val)
    dset3 = PyTorchDataset(y_val[:, 0], val_mesh, input_size, tau)
    valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
    n_val = len(valid_loader.dataset)
    epsilon = np.zeros((n_val, s), dtype=np.float32)

    for k in range(s):
        # dset2 = PyTorchDataset(y_sub[:, k], sub_mesh, input_size, tau)
        # subtrain_loader = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
        # dset3 = PyTorchDataset(y_val[:, k], val_mesh, input_size, tau)
        # valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)

        #print(f"Beginning component {k}")
        # model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)

        # # Training
        # optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        # scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

        # opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
        #     model, device, optimiser, scheduler, subtrain_loader, valid_loader,
        #     log_interval, patience, max_epochs, hidden_sizes)

        # print("Minimum validation loss: ", min_valid_loss)

        # with open(f"results/{model_name}.txt", "a") as file:
        #     file.write(f"\nComponent {k}\n")
        #     file.write(f"Optimum number of parameter update (opt_upd): {opt_upd}\n")
        #     file.write(f"Subtrain updates pr epoch (upd_epoch): {upd_epoch}\n")
        #     file.write(f"Minimum validation loss: {min_valid_loss}\n")

        # # Save learning curves
        # with open(f'results/validation_loss_{model_name}_{k}.npy', 'wb') as file:
        #     np.save(file, valid_loss)
        # with open(f'results/training_loss_{model_name}_{k}.npy', 'wb') as file:
        #     np.save(file, training_loss)

        # epsilon[:, k] = test(model, device, valid_loader, hidden_sizes)

        if k < s-1:
            dset2 = PyTorchDataset(y_sub[:, k], sub_mesh, input_size, tau)
            subtrain_loader = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
            dset3 = PyTorchDataset(y_val[:, k], val_mesh, input_size, tau)
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
                # file.write(f"Validation loss: {np.mean(valid_loss)}\n")
                # file.write(f"Training loss: {np.mean(training_loss)}\n")
                file.write(f"Minimum validation loss: {min_valid_loss}\n")
    
            # Save learning curves
            with open(f'results/validation_loss_{model_name}_{k}.npy', 'wb') as file:
                np.save(file, valid_loss)
            with open(f'results/training_loss_{model_name}_{k}.npy', 'wb') as file:
                np.save(file, training_loss)
    
            epsilon[:, k] = test(model, device, valid_loader, hidden_sizes)
            
        elif k == s-1:
            dset3 = PyTorchDataset(y_val[:, k], val_mesh, input_size, tau)
            valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
            print(f"Beginning component {k}")
            epsilon[:, k] = test_persistence(device, valid_loader)
        
        with open(f"results/{model_name}.txt", "a") as file:
                file.write(f"Validation MSE {np.mean(epsilon[:, k]**2)}\n")
                


    np.save(f"results/{model_name}_eps.npy", epsilon)
    tot_val_loss = np.mean(np.sum(epsilon, axis=1)**2)
    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"\nTotal validation loss: {tot_val_loss}")
