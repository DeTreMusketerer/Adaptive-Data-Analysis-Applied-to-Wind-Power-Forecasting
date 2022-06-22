"""
Created on Mon Apr 11 15:20:13 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, early stopping is done for LSTM models using the subtraining
data and the validation data.

Track changes:
    version 1.0: Basic implementation of neural network training script
                 for baseline LSTM model. (20/04/2022)
            1.1: Converting to only a validation script. (29/04/2022)
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Modules.NN_module import LSTM, GRU, PyTorchDataset, early_stopping


if __name__ == '__main__':
    # =============================================================================
    # Parameter specifications
    # =============================================================================
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = 100 # Maximum number of epochs.
    input_size = 3 # Number of samples in a training datapoint.
    tau = 12 # Number of samples we predict ahead.
    batch_size = 32
    learning_rate = 1e-03 # Initial learning rate.
    hidden_sizes = [128, 128, 128] # Number of hidden units in hidden layers.
    dropout_hidden = 0.3 # Dropout rate.
    gamma = 0.7 # Learning rate decay.
    log_interval = 600 # Interval for logging validation loss during early stopping.
    patience = 60 # Patience parameter for early stopping.
    Type = "LSTM" # Neural network model type, options "LSTM", "GRU".
    model_name = "LSTM001"

    # =============================================================================
    # Import data
    # =============================================================================
    y_sub = np.load("Data/subtraining_data.npy")
    sub_mesh = np.load("Data/subtraining_mesh_not_realtime.npy")

    y_val = np.load("Data/validation_data.npy")
    val_mesh = np.load("Data/validation_mesh_not_realtime.npy")

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
        file.write(f"Random seed: {seed}\n")

    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)

    # Training
    optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
    scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

    opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
        model, device, optimiser, scheduler, subtrain_loader, valid_loader,
        log_interval, patience, epochs, hidden_sizes)

    print("Minimum validation loss: ", min_valid_loss)

    with open(f"results/{model_name}.txt", "a") as file:
        file.write(f"Optimum number of parameter update (opt_upd): {opt_upd}\n")
        file.write(f"Subtrain updates pr epoch (upd_epoch): {upd_epoch}\n\n")
        file.write(f"Minimum validation loss: {min_valid_loss}")

    # Save learning curves
    with open(f'results/validation_loss_{model_name}.npy', 'wb') as file:
        np.save(file, valid_loss)
    with open(f'results/training_loss_{model_name}.npy', 'wb') as file:
        np.save(file, training_loss)

