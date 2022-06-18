"""
Created on Mon Apr 11 15:03:27 2022

Univariate-RNN on the danish wind power production.

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

This module implements:
    - Neural network training procedure using early stopping
    - LSTM neural network functionality
    - Data batch sampling.

Track changes:
    version 1.0: Base implementation. Adapted from 9th semester project.
                 Changed LSTM and GRU classes to allow varying number of layers.
                 Defined a PyTorchDataset_Realtime class for use with windows.
                 Also changed PyTorchDataset class to be similar in principle
                 to PyTorchDataset_Realtime. (19/04/2022)
            1.1: Changes test() function to output the raw error epsilon. (20/04/2022)
            1.2: Bugfix LSTM and GRU classes. (26/04/2022)
            1.3: Now in RealTime dataset if a component for a window is zeros,
                 then this datapoint is skipped.
            1.4: Fixed wrong handling of hidden states in LSTM. (10/05/2022)
"""


import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import Dataset


class LSTM(nn.Module):
    """
    Base class for the LSTM neural networks.
    """
    def __init__(self, input_size, hidden_sizes, dropout_hidden, device="cuda"):
        """
        Initialise the neural network structure.

        Inputs
        ----------
        input_size : int
            Size of the input to the neural network.
        hidden_sizes : list
            List containing the number of hidden units in each layer. The
            first entries are the number of hidden units in the consecutive
            LSTM layers and the last entry is the number of hidden units in
            the output dense layer.
        dropout_hidden : float
            Dropout rate for hidden units.
        device : str, optional
            Options are 'cuda' or 'cpu'. The default is 'cuda'.

        Parameters
        ----------
        layers : list
            List containing the layers of the neural network.
        """
        super(LSTM, self).__init__()
        self.dropout_hidden = nn.Dropout(dropout_hidden)
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTM(input_size, hidden_sizes[0], 1))
        for i in range(len(hidden_sizes[1:])):
            self.layers.append(nn.LSTM(hidden_sizes[i], hidden_sizes[i+1], 1))
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.hidden_sizes = hidden_sizes
        self.device = device

    def forward(self, x, h_old, s_old):
        x = x.unsqueeze(1)
        x = x.float()
        new_h = torch.zeros(len(self.hidden_sizes), max(self.hidden_sizes)).to(self.device)
        new_s = torch.zeros(len(self.hidden_sizes), max(self.hidden_sizes)).to(self.device)
        for i, layer in enumerate(self.layers[:-1]):
            h_in = h_old[i, :self.hidden_sizes[i]].unsqueeze(0).unsqueeze(0)
            s_in = s_old[i, :self.hidden_sizes[i]].unsqueeze(0).unsqueeze(0)
            x, (h, s) = layer(x, (h_in.detach(), s_in.detach()))
            new_h[i, :self.hidden_sizes[i]] = h[0, 0, :]
            new_s[i, :self.hidden_sizes[i]] = s[0, 0, :]
            x = self.dropout_hidden(x)
        x = self.layers[-1](x)
        return x, new_h, new_s


# class LSTM_joined(nn.Module):
#     """
#     Base class for the LSTM neural networks.
#     """
#     def __init__(self, input_size, hidden_sizes, dropout_hidden, nr_components, device="cuda"):
#         """
#         Initialise the neural network structure.

#         Inputs
#         ----------
#         input_size : int
#             Size of the input to the neural network.
#         hidden_sizes : list
#             List containing the number of hidden units in each layer. The
#             first entries are the number of hidden units in the consecutive
#             LSTM layers and the last entry is the number of hidden units in
#             the output dense layer.
#         dropout_hidden : float
#             Dropout rate for hidden units.

#         Parameters
#         ----------
#         layers : list
#             List containing the layers of the neural network.
#         """
#         super(LSTM_joined, self).__init__()
#         self.device = device
#         self.nr_components = nr_components
#         self.dropout_hidden = nn.Dropout(dropout_hidden)
#         self.nr_layers = len(hidden_sizes)
#         self.all_layers = nn.ModuleList()
#         for k in range(self.nr_components):
#             self.all_layers.append(nn.LSTM(input_size, hidden_sizes[0], 1))
#             for i in range(len(hidden_sizes[1:])):
#                 self.all_layers.append(nn.LSTM(hidden_sizes[i], hidden_sizes[i+1], 1))
#             self.all_layers.append(nn.Linear(hidden_sizes[-1], 1))

#     def forward(self, x_all):
#         batch_size = x_all.size(dim=0)
#         tot_pred = torch.zeros(batch_size, self.nr_components).to(self.device)
#         for k in range(self.nr_components):
#             x = x_all[:, k, :].unsqueeze(1)
#             x = x.float()
#             for layer in self.all_layers[k*(self.nr_layers+1):k*(self.nr_layers+1)+self.nr_layers]:
#                 x, _ = layer(x)
#                 x = self.dropout_hidden(x)
#             tot_pred[:, k] = (self.all_layers[k*(self.nr_layers+1)+self.nr_layers])(x).squeeze()
#         return torch.sum(tot_pred, dim=1)


class GRU(nn.Module):
    """
    Base class for the GRU neural networks.
    """
    def __init__(self, input_size, hidden_sizes, dropout_hidden):
        """
        Initialise the neural network structure.

        Inputs
        ----------
        input_size : int
            Size of the input to the neural network.
        hidden_sizes : list
            List containing the number of hidden units in each layer. The
            first entries are the number of hidden units in the consecutive
            GRU layers and the last entry is the number of hidden units in
            the output dense layer.
        dropout_hidden : float
            Dropout rate for hidden units.

        Parameters
        ----------
        layers : list
            List containing the layers of the neural network.
        """
        super(GRU, self).__init__()
        self.dropout_hidden = nn.Dropout(dropout_hidden)
        self.layers = nn.ModuleList()
        self.layers.append(nn.GRU(input_size, hidden_sizes[0], 1))
        for i in range(len(hidden_sizes[1:])):
            self.layers.append(nn.GRU(hidden_sizes[i], hidden_sizes[i+1], 1))
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.hidden_sizes = hidden_sizes
        self.hidden = torch.zeros(len(hidden_sizes), max(hidden_sizes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.float()
        for i, layer in enumerate(self.layers[:-1]):
            x, h = layer(x, self.hidden[i, :self.hidden_sizes[i]].unsqueeze(0))
            self.hidden[i, :self.hidden_sizes[i]] = h
            x = self.dropout_hidden(x)
        x = self.layers[-1](x)
        return x


class PyTorchDataset(Dataset):
    """
    PyTorch dataset. Used to load datapoints during neural network training
    and testing.
    """
    def __init__(self, y, mesh, input_size: int=36, tau: int=12):
        """
        Inputs
        ------
        y : ndarray, size=(n,)
            The dataset.
        mesh : ndarray, size=(n,)
            Boolean mesh to keep track of missing data points.
        input_size : int, optional
            Size of the input to the neural network. The default is 36.
        tau : int, optional
            Number of samples we predict ahead. The default is 12.

        Parameters
        ----------
        endpoint : int
            Index we are trying to predict compared to the input index as
            defined in the __getitem__() function,
            i.e. endpoint = input_size + tau - 1.
        """
        self.y = y
        self.input_size = input_size
        self.idx_power = self.NN_idx(mesh, tau)
        self.tau = tau
        self.endpoint = self.input_size + self.tau - 1

    def NN_idx(self, mesh, tau):
        """
        Map indices i = 1,\dots,n to indices excluding points that cannot
        be used for forecasting due to missing data.
    
        Inputs
        -------
        mesh : ndarray, size=(n,)
            The mesh as prepared in the Data_Preparation.py script.
        tau : int
            Indices for \tau-ahead forecasts

        Returns
        -------
        idx_power : ndarray, size(n_new,)
            Index array mapping i=1,\dots,n to indices j=1,\dots,n_new
            where n_new < n and n_new depends on the missing data.
        """
        n = len(mesh)
        idx_power_list = list()
        for i in range(n-tau-self.input_size):
            if (mesh[i:i+self.input_size+tau+1] == 1).all():
                idx_power_list.append(i)
        idx_power = np.array(idx_power_list, dtype=np.int32)
        return idx_power

    def __len__(self):
        "Number of datapoints in the dataset."
        return len(self.idx_power)

    def __getitem__(self, idx):
        """
        Retrieve a datapoint from the dataset. The input starts at index_p
        defined via idx.
        """
        index_p = self.idx_power[idx]
        P_history = self.y[index_p:index_p + self.input_size].astype('float32')
        P_tau = self.y[index_p + self.endpoint].astype('float32')
        data = P_history
        datapoint = torch.from_numpy(data)
        target = torch.from_numpy(np.array(P_tau)).type(torch.Tensor)
        sample = (datapoint, target)
        return sample


class PyTorchDataset_RealTime(Dataset):
    """
    PyTorch dataset. Used to load datapoints during neural network training
    and testing for real-time models.
    """
    def __init__(self, y, mesh, xi: int=12, input_size: int=36, tau: int=12):
        """
        Inputs
        ------
        y : ndarray, size=(n, q)
            The dataset.
        mesh : ndarray, size=(n,)
            Boolean mesh to keep track of missing data points.
        xi : int, optional
            Window shift to avoid end effects in target. The default is 12.
        input_size : int, optional
            Size of the input to the neural network. The default is 36.
        tau : int, optional
            Number of samples we predict ahead. The default is 12.
        """
        self.y = y
        self.xi = xi
        self.tau = tau
        self.n, self.q = np.shape(y)
        self.input_size = input_size
        self.idx_power = self.NN_idx(mesh, tau)

    def NN_idx(self, mesh, tau):
        """
        Map indices i = 1,\dots,n to indices excluding points that cannot
        be used for forecasting due to missing data.
    
        Inputs
        -------
        mesh : ndarray, size=(n,)
            The mesh as prepared in the Data_Preparation.py script.
        tau : int
            Indices for \tau-ahead forecasts
    
        Parameters
        ----------
        shift : int
            Defined as self.xi and gives the shift we do in order to pick a target
            that is not at the boundary of its window.
    
        Returns
        -------
        idx_power : ndarray, size(n_new,)
            Index array mapping i=1,\dots,n to indices j=1,\dots,n_new
            where n_new < n and n_new depends on the missing data.
        """
        n = len(mesh)
        shift = self.xi
        idx_power_list = list()
        for i in range(n-tau-shift):
            if (mesh[i:i+tau+shift+1] == 1).all():
                idx_power_list.append(i)
        idx_power = np.array(idx_power_list, dtype=np.int32)
        return idx_power

    def update_xi(self, xi):
        self.xi = xi

    def __len__(self):
        "Number of datapoints in the dataset."
        return len(self.idx_power)

    def __getitem__(self, idx):
        """
        Retrieve a datapoint from the dataset. The input to the neural
        network is the indices t-input_size+1, ..., t and the point being
        forecasted is t+\tau where idx is mapped to a time t by
        window_mesh.
        """
        window_index = self.idx_power[idx]
        P_history = self.y[window_index, -self.input_size:].astype('float32')
        target_window_index = window_index+self.tau+self.xi
        P_tau = self.y[target_window_index, self.q-1-self.xi].astype('float32')
        data = P_history
        datapoint = torch.from_numpy(data)
        target = torch.from_numpy(np.array(P_tau)).type(torch.Tensor)
        sample = (datapoint, target)
        return sample


# class PyTorchDataset_RealTime_joined(Dataset):
#     """
#     PyTorch dataset. Used to load datapoints during neural network training
#     and testing for real-time models.
#     """
#     def __init__(self, y, power, mesh, input_size: int=36, tau: int=12):
#         """
#         Inputs
#         ------
#         y : ndarray, size=(n, J, q)
#             The dataset.
#         power : ndarray, size=(n,)
#             The power data.
#         mesh : ndarray, size=(n,)
#             Boolean mesh to keep track of missing data points.
#         input_size : int, optional
#             Size of the input to the neural network. The default is 36.
#         tau : int, optional
#             Number of samples we predict ahead. The default is 12.
#         """
#         self.y = y
#         self.power = power
#         self.tau = tau
#         self.n, self.J, self.q = np.shape(y)
#         self.input_size = input_size
#         self.idx_power = self.NN_idx(mesh, tau)

#     def NN_idx(self, mesh, tau):
#         """
#         Map indices i = 1,\dots,n to indices excluding points that cannot
#         be used for forecasting due to missing data.
    
#         Inputs
#         -------
#         mesh : ndarray, size=(n,)
#             The mesh as prepared in the Data_Preparation.py script.
#         tau : int
#             Indices for \tau-ahead forecasts
    
#         Returns
#         -------
#         idx_power : ndarray, size(n_new,)
#             Index array mapping i=1,\dots,n to indices j=1,\dots,n_new
#             where n_new < n and n_new depends on the missing data.
#         """
#         n = len(mesh)
#         idx_power_list = list()
#         for i in range(n-tau-1):
#             if (mesh[i:i+tau+1] == 1).all():
#                 idx_power_list.append(i)
#         idx_power = np.array(idx_power_list, dtype=np.int32)
#         return idx_power

#     def __len__(self):
#         "Number of datapoints in the dataset."
#         return len(self.idx_power)

#     def __getitem__(self, idx):
#         """
#         Retrieve a datapoint from the dataset. The input to the neural
#         network is the indices t-input_size+1, ..., t and the point being
#         forecasted is t+\tau where idx is mapped to a time t by
#         window_mesh.
#         """
#         window_index = self.idx_power[idx]
#         P_history = self.y[window_index, :, -self.input_size:].astype('float32')
#         P_tau = self.power[window_index+self.tau+self.q-1].astype('float32')
#         data = P_history
#         datapoint = torch.from_numpy(data)
#         target = torch.from_numpy(np.array(P_tau)).type(torch.Tensor)
#         sample = (datapoint, target)
#         return sample


def test(model, device, test_loader, hidden_sizes):
    """
    Tests the trained model in terms of MSE and NMAE

    Parameters
    ----------
    model : Pytorch model class
    device : device
    test_loader : Dataloader
        Dataloader for the test set.
    hidden_sizes : list
        List of integers giving the hidden sizes for the neural network model.
        This is used for the hidden states in the LSTM.

    Returns
    -------
    epsilon : ndarray, size=(n,)
        The test error.
    """
    model.eval()
    n = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    epsilon = np.zeros(n, dtype = 'float32')
    batches, remainder = np.divmod(n, batch_size)
    #MSE_loss = 0
    #NMAE_loss = 0
    h = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    s = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    with torch.no_grad():
        for i, (test, test_target) in enumerate(test_loader):
            test, test_target = test.to(device), test_target.to(device)
            out_test, h, s = model(test, h, s)
            output_test = out_test.squeeze()
            if remainder != 0:
                if i < batches-1:
                    epsilon[i*batch_size: (i+1)*batch_size] = (output_test - test_target).cpu().numpy()
                else:
                    epsilon[-remainder:] = (output_test - test_target).cpu().numpy()[:remainder]
            elif remainder == 0:
                epsilon[i*batch_size: (i+1)*batch_size] = (output_test - test_target).cpu().numpy()
            #MSE_loss += F.mse_loss(output_test, test_target, reduction='sum').item()  # sum up batch loss
            #NMAE_loss += torch.sum(torch.absolute(test_target - output_test)/y_max).item()
    #MSE_loss /= len(test_loader.dataset)
    #NMAE_loss /= len(test_loader.dataset)
    model.train()
    return epsilon

def test_pred_zero(device, test_loader):
    """
    Tests the trained model in terms of MSE and NMAE
    predicting zero at every step.

    Parameters
    ----------
    device : device
    test_loader : Dataloader
        Dataloader for the test set.

    Returns
    -------
    epsilon : ndarray, size=(n,)
        The test error.
    """
    n = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    epsilon = np.zeros(n, dtype = 'float32')
    batches, remainder = np.divmod(n, batch_size)
    #MSE_loss = 0
    #NMAE_loss = 0
    with torch.no_grad():
        for i, (test, test_target) in enumerate(test_loader):
            test, test_target = test.to(device), test_target.to(device)
            if remainder != 0:
                if i < batches-1:
                    epsilon[i*batch_size: (i+1)*batch_size] = (- test_target).cpu().numpy()
                else:
                    epsilon[-remainder:] = (- test_target).cpu().numpy()[:remainder]
            elif remainder == 0:
                epsilon[i*batch_size: (i+1)*batch_size] = (- test_target).cpu().numpy()
            #MSE_loss += F.mse_loss(output_test, test_target, reduction='sum').item()  # sum up batch loss
            #NMAE_loss += torch.sum(torch.absolute(test_target - output_test)/y_max).item()
    #MSE_loss /= len(test_loader.dataset)
    #NMAE_loss /= len(test_loader.dataset)
    return epsilon

def test_persistence(device, test_loader):
    """
    Tests the trained model in terms of MSE and NMAE using persistence.

    Parameters
    ----------
    device : device
    test_loader : Dataloader
        Dataloader for the test set.

    Returns
    -------
    epsilon : ndarray, size=(n,)
        The test error.
    """
    n = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    epsilon = np.zeros(n, dtype = 'float32')
    batches, remainder = np.divmod(n, batch_size)
    #MSE_loss = 0
    #NMAE_loss = 0
    with torch.no_grad():
        for i, (test, test_target) in enumerate(test_loader):
            test, test_target = test.to(device), test_target.to(device)
            if remainder != 0:
                if i < batches-1:
                    epsilon[i*batch_size: (i+1)*batch_size] = (test[:, -1] - test_target).cpu().numpy()
                else:
                    epsilon[-remainder:] = (test[:, -1] - test_target).cpu().numpy()[:remainder]
            elif remainder == 0:
                epsilon[i*batch_size: (i+1)*batch_size] = (test[:, -1] - test_target).cpu().numpy()
            #MSE_loss += F.mse_loss(output_test, test_target, reduction='sum').item()  # sum up batch loss
            #NMAE_loss += torch.sum(torch.absolute(test_target - output_test)/y_max).item()
    #MSE_loss /= len(test_loader.dataset)
    #NMAE_loss /= len(test_loader.dataset)
    return epsilon

def validation(model, device, valid_loader, hidden_sizes):
    """
    Evaluates the validation set. This is done during early stopping.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    valid_loader : Dataloader
        Dataloader for the validation set.
    hidden_sizes : list
        List of integers giving the hidden sizes for the neural network model.
        This is used for the hidden states in the LSTM.
    
    Returns
    -------
    valid_loss : float
        The validation loss in MSE.
    """
    model.eval()
    valid_loss = 0
    h = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    s = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    with torch.no_grad():
        for valid, valid_target in valid_loader:
            valid, valid_target = valid.to(device), valid_target.to(device)
            out_valid, h, s = model(valid, h, s)
            output_valid = out_valid.squeeze()
            valid_loss += F.mse_loss(output_valid, valid_target, reduction='sum').item()  # sum up batch loss
    valid_loss /= len(valid_loader.dataset)
    model.train()
    return valid_loss


def early_stopping(model, device, optimiser, scheduler, subtrain_loader,
                   valid_loader, log_interval, patience, max_epochs, hidden_sizes):
    """
    Determines the number of parameter updates which should be performed
    during training using early stopping.

    Parameters
    ----------
    model : PyTorch model class
    device : device
    optimiser : PyTorch optimiser
    scheduler : PyTorch scheduler
    subtrain_loader : Dataloder
        Subtrain dataset.
    valid_loader : Dataloader
        Validation dataset.
    log_interval: int
        Time between prints of performance.
    patience : int
        Patience parameter in early stopping.
    max_epochs : int
        Max number of epochs to train.
    hidden_sizes : list
        List of integers giving the hidden sizes for the neural network model.
        This is used for the hidden states in the LSTM.

    Returns
    -------
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.
    """
    updates_counter = 0
    min_valid_loss = 0
    no_increase_counter = 0
    optim_updates = 0
    updates_pr_pretrain_epoch = len(subtrain_loader)
    valid_loss_list = []
    training_loss_list = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        interval_loss = 0
        h = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
        s = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
        for batch_idx, (data, target) in enumerate(subtrain_loader):
            data, target = data.to(device), target.to(device)

            target = target.squeeze()
            optimiser.zero_grad()
            out, h, s = model(data, h, s)
            output = out.squeeze()
            loss = F.mse_loss(output, target, reduction='mean')
            loss.backward()
            optimiser.step()

            interval_loss += loss.item()
            if batch_idx % log_interval == 0 and batch_idx != 0:
                valid_loss = validation(model, device, valid_loader, hidden_sizes)
                valid_loss_list.append(valid_loss)
                training_loss_list.append(interval_loss/log_interval)

                if min_valid_loss == 0:
                    min_valid_loss = valid_loss
                elif valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    optim_updates = updates_counter
                    no_increase_counter = 0
                else:
                    no_increase_counter += 1

                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tMin. Val. Loss: {:.6f}\tVal. Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(subtrain_loader.dataset),
                    interval_loss/log_interval, min_valid_loss, valid_loss))
                interval_loss = 0

                if no_increase_counter == patience:
                    return optim_updates, updates_pr_pretrain_epoch, valid_loss_list, training_loss_list, min_valid_loss

            updates_counter += 1
        scheduler.step()
        print('')
        if no_increase_counter == patience: # Det her sker vel aldrig?
            break
    return optim_updates, updates_pr_pretrain_epoch, valid_loss_list, training_loss_list, min_valid_loss


def early_stopping_retrain(model, device, train_loader, optimiser, epoch,
                           optim_updates, updates_counter, scheduler,
                           updates_pr_pretrain_epoch, log_interval,
                           hidden_sizes):
    """
    Re-trains the neural network after early stopping pre-training.
    The learning rate is decayed after each updates_pr_pretrain_epoch 
    parameter updates and training is done for optim_updates
    parameter updates.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    train_loader : Dataloader
        Training dataset.
    optimiser : PyTorch optimiser
    epoch : int
        The current training epoch.
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_counter : int
        Counter for the number of parameter updates.
    scheduler : PyTorch scheduler
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.
    log_interval : int 
        time between prints of performance.
    hidden_sizes : list
        List of integers giving the hidden sizes for the neural network model.
        This is used for the hidden states in the LSTM.
        
    Returns
    -------
    updates_counter : int
        Counter for the number of parameter updates.
    """
    model.train()
    interval_loss = 0
    h = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    s = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target = target.squeeze()
        optimiser.zero_grad()
        out, h, s = model(data, h, s)
        output = out.squeeze()
        loss = F.mse_loss(output, target)
        #optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        interval_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), interval_loss/log_interval))
            interval_loss = 0

        if updates_counter == optim_updates:
            return updates_counter
        updates_counter += 1
        if updates_counter % updates_pr_pretrain_epoch == 0:
            scheduler.step()
    return updates_counter
