import numpy as np
import matplotlib.pyplot as plt
import torch
import Modules.HilbertHuangTransform as HHT
import pandas as pd
import matplotlib.dates as mdates
from NN_module import LSTM, PyTorchDataset_RealTime, PyTorchDataset



def Forecast(model, device, test_loader, hidden_sizes):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    test_loader : TYPE
        DESCRIPTION.
    hidden_sizes : TYPE
        DESCRIPTION.

    Returns
    -------
    output_array : TYPE
        DESCRIPTION.

    """
    model.eval()
    n = len(test_loader.dataset)
    output_array = np.zeros(n)
    h = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    s = torch.zeros(len(hidden_sizes), max(hidden_sizes)).to(device)
    batches, remainder = np.divmod(n, batch_size)
    with torch.no_grad():
        with torch.no_grad():
            for i, (test, test_target) in enumerate(test_loader):
                    test, test_target = test.to(device), test_target.to(device)
                    out_test, h, s = model(test, h, s)
                    output_test = out_test.squeeze()  
                    if remainder != 0:
                        if i < batches-1:
                            output_array[i*batch_size: (i+1)*batch_size] = output_test.cpu().numpy()
                        else:
                            output_array[-remainder:] = output_test.cpu().numpy()[:remainder]
                    elif remainder == 0:
                        output_array[i*batch_size: (i+1)*batch_size] = output_test.cpu().numpy()
    return output_array


def Forecast_persistence(device, test_loader):
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
    output_array = np.zeros(n, dtype = 'float32')
    batches, remainder = np.divmod(n, batch_size)
    with torch.no_grad():
        for i, (test, test_target) in enumerate(test_loader):
            test = test.to(device)
            if remainder != 0:
                if i < batches-1:
                    output_array[i*batch_size: (i+1)*batch_size] = test[:, -1].cpu().numpy()
                else:
                    output_array[-remainder:] = test[:, -1].cpu().numpy()[:remainder]
            elif remainder == 0:
                output_array[i*batch_size: (i+1)*batch_size] = test[:, -1].cpu().numpy()
    return output_array


model_name_PDE = "PDE-EMD-Live-LSTM089"

# =============================================================================
# Read model settings
# =============================================================================
with open(f"results/{model_name_PDE}.txt", "r") as file:
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
s = eval(s)
y_test = np.load(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
test_mesh = np.load(f"Data/test_mesh_q{q}.npy")

opt_upd = np.zeros(s+1)
upd_epoch = np.zeros(s+1)
with open(f"results/{model_name_PDE}.txt", "r") as file:
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dset = PyTorchDataset_RealTime(y_test[:, 0, :], test_mesh, xi, input_size, tau)
test_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle = False)
n_test = len(test_loader.dataset)
forecast_array = np.zeros(n_test, dtype=np.float32)

#PDE
for k in range(s+1):
    if upd_epoch[k] != 0:
        model = LSTM(input_size, hidden_sizes, dropout_hidden).to(device)
        dset = PyTorchDataset_RealTime(y_test[:, k, :], test_mesh, xi, input_size, tau)
        test_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle = False)
        model.load_state_dict(torch.load(f"models/{model_name_PDE}_{k}.pt", map_location=device))
        forecast_array += Forecast(model, device, test_loader, hidden_sizes)
        print(forecast_array[1])
np.save(f"Data/forecast_array_{model_name_PDE}.npy", forecast_array)



#EMD
input_size_EMD = 3
hidden_sizes_EMD = [128,128,128]
dropout_EMD = 0.1
test_mesh_not_realtime = np.load("Data/test_mesh_not_realtime.npy")
model_name_EMD = "EMD-LSTM128"
y_test_EMD = np.load("Data/EMD_full_test_data.npy")
dset2 = PyTorchDataset(y_test_EMD[:, 0], test_mesh_not_realtime, input_size_EMD, tau)
test_loader_EMD = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
n_test_EMD = len(test_loader_EMD.dataset)
forecast_array_EMD = np.zeros(n_test_EMD, dtype=np.float32)
model_EMD = LSTM(input_size_EMD, hidden_sizes_EMD, dropout_EMD).to(device)
for k in range(20):
    if k != 19:
        dset2 = PyTorchDataset(y_test_EMD[:, k], test_mesh_not_realtime, input_size_EMD, tau)
        test_loader_EMD = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
        model_EMD.load_state_dict(torch.load(f"models/{model_name_EMD}_{k}.pt", map_location=device))
        forecast_array_EMD += Forecast(model_EMD, device, test_loader_EMD, hidden_sizes_EMD)
    else:
        dset2 = PyTorchDataset(y_test_EMD[:, k], test_mesh_not_realtime, input_size_EMD, tau)
        test_loader_EMD = torch.utils.data.DataLoader(dset2, batch_size, shuffle = False)
        forecast_array_EMD += Forecast_persistence(device, test_loader_EMD)
np.save(f"Data/forecast_array_EMD_{model_name_EMD}.npy", forecast_array_EMD)



test_data = np.load("Data/test_data.npy")
model_name_EMD = "EMD-LSTM128"
EMD_forecast = np.load(f"Data/forecast_array_EMD_{model_name_EMD}.npy")
model_name_PDE = "PDE-EMD-Live-LSTM089"
PDE_forecast = np.load(f"Data/forecast_array_{model_name_PDE}.npy")
t_start = 2000-104
t_end = 2288-104

HHT.plot_style()
x = pd.date_range("2021-10-12 00:00:00", "2021-10-12 23:55:00", freq="5min")
fig, ax = plt.subplots(1, 1)
ax.plot(x,test_data[t_start:t_end], label = "Power production")
ax.plot(x,EMD_forecast[t_start-15:t_end-15], label = "Offline EMD-LSTM forecast") # Højde for p også er der en times offset mere
ax.plot(x,PDE_forecast[t_start-289:t_end-289], label = "PDE-EMD-LSTM forecast") # man skal tage højde for q og p
locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.legend()
plt.ylabel("Power Production [MW]")
plt.savefig(f"figures/Forecast_{t_start}_{t_end}.png", dpi = 600)
plt.show()