"""
Created on Tue Apr 19 15:42:06 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script, early stopping is done for PDE-EMD-LSTM models using the
subtraining data and the validation data in an online setup.

Track changes:
    version 1.0: Basic adaptation of the online training script. This is
                 still not completely ready for use. (20/04/2022)
    version 1.1: The script can run correctly.
"""


import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from NN_module import LSTM, PyTorchDataset_RealTime, early_stopping, test, test_pred_zero


if __name__ == '__main__':
    # =============================================================================
    # Parameter specifications
    # =============================================================================
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = 200 # Maximum number of epochs.
    input_size = 1 # Number of samples in a training dataset.
    tau = 12 # Number of samples we predict ahead.
    batch_size = 32
    learning_rate = 0.0007 # Initial learning rate.
    hidden_sizes = [128, 256, 128] # Number of hidden units in hidden layers.
    dropout_hidden = 0.05 # Dropout rate.
    gamma = 0.7 # Learning rate decay.
    log_interval = 600 # Interval for logging validation loss during early stopping.
    patience = 200 # Patience parameter for early stopping.
    q = 288 # Window length
    xi = 2 # Shift of window for target
    T = 6
    s = 3
    boundary = "Neumann_0"
    Type = "LSTM" # Neural network model type, options "LSTM", "GRU".
    data_type = "normal"
    model_name = "PDE-EMD-Live-LSTM001"


    # =============================================================================
    # Import data
    # =============================================================================
    if data_type == "IMF-unified":
        y_train = np.load(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy")        
    elif data_type == "normal":
        y_train = np.load(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
    else:
        print("Data type not supported")
    train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
    power_sub = np.load("Data/subtraining_data.npy")
    n_sub = np.shape(power_sub)[0]

    y_sub = y_train[:n_sub, :, :]
    sub_mesh = train_mesh[:n_sub]

    y_val = y_train[n_sub:, :, :]
    val_mesh = train_mesh[n_sub:]
    n_val = np.shape(y_val)[0]


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
        file.write(f"Random seed: {seed}\n")
        file.write(f"T: {T}\n")
        file.write(f"s: {s}\n")
        file.write(f"boundary: {boundary}\n")
        file.write(f"Data type: {data_type}\n")


    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    epsilon = np.zeros((n_val-q-tau-xi, s+1))
    # if s == 3 or data_type == "IMF-unified": # this is quite specific for unified data
    #     s = 3
    #     pred_zero = [0,0,1,1]
    # else:
    #     s = 4
    #     pred_zero = [0,0,1,1,1]
    pred_zero = [0,0,1,1,1]
    for k in range(0,s+1):
        dset3 = PyTorchDataset_RealTime(y_val[:,k,:].reshape(n_val,q), val_mesh, xi, input_size, tau)
        valid_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
        if pred_zero[k] == 1:
            print(f"Beginning component {k}")
            model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)
            
            dset2 = PyTorchDataset_RealTime(y_sub[:,k,:].reshape(n_sub,q), sub_mesh, xi, input_size, tau)
            subtrain_loader = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
            
            # Training
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
        
            opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
                model, device, optimiser, scheduler, subtrain_loader, valid_loader,
                log_interval, patience, epochs, hidden_sizes)
        
            print("Minimum validation loss: ", min_valid_loss)
        
            with open(f"results/{model_name}.txt", "a") as file:
                file.write(f"\nComponent {k}\n")
                file.write(f"Optimum number of parameter update (opt_upd): {opt_upd}\n")
                file.write(f"Subtrain epochs (upd_epoch): {upd_epoch}\n")
                file.write(f"Minimum validation loss: {min_valid_loss}\n")
                file.write(f"Minimum training loss: {np.min(training_loss)}\n")
                file.write(f"Component energy: {np.mean(y_val[:-288, k, :]**2)}\n")
                file.write(f"Component variance: {np.mean((y_val[:-288, k, :] - np.mean(y_val[:-288, k, :]))**2)}\n")
                file.write(f"Xi: {np.mean((y_val[:-288, k, -xi-1] - np.mean(y_val[:-288, k, -xi-1]))**2)}\n")
        
            # Save learning curves
            with open(f'results/validation_loss_{model_name}_{k}.npy', 'wb') as file:
                np.save(file, valid_loss)
            with open(f'results/training_loss_{model_name}_{k}.npy', 'wb') as file:
                np.save(file, training_loss)
        
            epsilon[:, k] = test(model, device, valid_loader, hidden_sizes)
        
        else:
            epsilon[:, k] = test_pred_zero(device, valid_loader)
            min_valid_loss = np.sum(epsilon[:,k]**2)/len(epsilon)
            with open(f"results/{model_name}.txt", "a") as file:
                file.write(f"\nComponent {k}\n")
                file.write("Optimum number of parameter update 0: 0\n")
                file.write("Subtrain epochs 0: 0\n")
                file.write(f"Minimum validation loss: {min_valid_loss}\n")
                file.write("Minimum training loss: None\n")
                file.write(f"Component energy: {np.mean(y_val[:-288, k, :]**2)}\n")
                file.write(f"Component variance: {np.mean((y_val[:-288, k, :] - np.mean(y_val[:-288, k, :]))**2)}\n")
                file.write(f"Xi: {np.mean((y_val[:-288, k, -xi-1] - np.mean(y_val[:-288, k, -xi-1]))**2)}\n")
            
    np.save(f"results/Validation_epsilon_{model_name}.npy", epsilon)
    MSE_val = np.mean(np.sum(epsilon, axis = 1)**2)
    print(MSE_val)
    