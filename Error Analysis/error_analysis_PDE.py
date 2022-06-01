import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import HilbertHuangTransform as HHT
import matplotlib.dates as mdates

#%% Test
model_number = "089"
epsilon_test = np.load(f"results/test_error_PDE-EMD-Live-LSTM{model_number}.npy")
MSE_test = np.mean(np.sum(epsilon_test, axis = 1)**2)
CAP = np.load("Data/DK1-1_Capacity.npy")/1000
NRMSE_test = np.sqrt(np.mean(np.sum((epsilon_test), axis = 1)**2/(CAP**2)))
NMAE_test = np.mean(np.abs(np.sum(epsilon_test/CAP, axis = 1)))
NBIAS_test = np.mean(np.sum(epsilon_test, axis=1)/CAP)
n_test,s = np.shape(epsilon_test)

component_MSE_test = np.zeros(s)
for k in range(s):
    component_MSE_test[k] = np.mean(epsilon_test[:,k]**2)

#%% Validation
model_number = "089"
epsilon_val = np.load(f"results/Validation_epsilon_PDE-EMD-Live-LSTM{model_number}.npy")
MSE_val = np.mean(np.sum(epsilon_val, axis = 1)**2)
CAP = np.load("Data/DK1-1_Capacity.npy")/1000
NRMSE_val = np.sqrt(np.mean(np.sum((epsilon_val), axis = 1)**2/(CAP**2)))
NMAE_val = np.mean(np.abs(np.sum(epsilon_val/CAP, axis = 1)))
NBIAS_val = np.mean(np.sum(epsilon_val, axis=1)/CAP)
n_val,s = np.shape(epsilon_val)

component_MSE_val = np.zeros(s)
for k in range(s):
    component_MSE_val[k] = np.mean(epsilon_val[:,k]**2)

#%%Percentile
methods = ["error_EMD-LSTM128", "test_error_FFT-NMP-EMD-LSTM002", "test_error_EMD-Live-LSTM057",
            "test_error_FFT-NMP-EMD-Live-LSTM002", "test_error_PDE-EMD-Live-LSTM089"]
percentile = 0.95
CAP = np.load("Data/DK1-1_Capacity.npy")/1000

model = methods[1]
epsilon_test = np.load(f"results/{model}.npy")
epsilon_test_entry = np.sum(epsilon_test, axis = 1)/CAP
n_test = len(epsilon_test_entry)
eps_sort = np.sort(abs(epsilon_test_entry))
sample = int(n_test*percentile)
zeta = eps_sort[sample]

## LSTM
epsilon_LSTM = np.load("results/test_error_LSTM032.npy")/CAP
eps_sort_LSTM = np.sort(abs(epsilon_LSTM))
n_test = len(eps_sort_LSTM)
sample = int(n_test*percentile)
zeta_LSTM = eps_sort_LSTM[sample]

## AR
epsilon_AR = np.load("results/AR_error.npy")
eps_12 = epsilon_AR[11,:,0]/CAP
eps_sort_AR = np.sort(abs(eps_12))
n_test = len(eps_sort_AR)
sample = int(n_test*percentile)
zeta_AR = eps_sort_AR[sample]

#%% Reverse Energint
CAP = np.load("Data/DK1-1_Capacity.npy")/1000
Energinet_MSE = (0.0465**2)*(CAP**2)

#%% KDE plot
methods = ["error_EMD-LSTM128", "test_error_FFT-NMP-EMD-LSTM002", "AR_error",
           "test_error_LSTM032", "test_error_EMD-Live-LSTM057",
           "test_error_FFT-NMP-EMD-Live-LSTM002", "test_error_PDE-EMD-Live-LSTM089"]
edge = 0.16*100
points = 500
x = np.linspace(-edge, edge, points)
CAP = np.load("Data/DK1-1_Capacity.npy")/1000
HHT.plot_style()

#EMD
eps_EMD = np.load(f"results/{methods[0]}.npy")
summed_EMD = np.sum(eps_EMD, axis = 1)
kde_EMD = ss.gaussian_kde(summed_EMD/CAP*100)
plt.plot(x, kde_EMD(x), label="Offline EMD-LSTM")

# # FFT
# eps_FFT = np.load(f"results/{methods[1]}.npy")
# summed_FFT = np.sum(eps_FFT, axis = 1)
# kde_FFT = ss.gaussian_kde(summed_FFT)
# plt.plot(x, kde_FFT(x), label="offline FFT-NMP-EMD-LSTM")

# # AR
# eps_AR = np.load(f"results/{methods[2]}.npy")
# eps_12 = eps_AR[11,:,0]
# kde_AR = ss.gaussian_kde(eps_12)
# plt.plot(x, kde_AR(x), label="AR")

# LSTM
eps_LSTM = np.load(f"results/{methods[3]}.npy")
kde_LSTM = ss.gaussian_kde(eps_LSTM/CAP*100)
plt.plot(x, kde_LSTM(x), label="LSTM")

# # EMD-Live
# eps_EMD_L = np.load(f"results/{methods[4]}.npy")
# summed_EMD_L = np.sum(eps_EMD_L, axis = 1)
# kde_EMD_L = ss.gaussian_kde(summed_EMD_L)
# plt.plot(x, kde_EMD_L(x), label="EMD-LSTM")

# # FFT_Live
# eps_FFT_L = np.load(f"results/{methods[5]}.npy")
# summed_FFT_L = np.sum(eps_FFT_L, axis = 1)
# kde_FFT_L = ss.gaussian_kde(summed_FFT_L)
# plt.plot(x, kde_FFT_L(x), label="FFT-NMP-EMD-LSTM")

#PDE
eps_PDE = np.load(f"results/{methods[6]}.npy")
summed_PDE = np.sum(eps_PDE, axis = 1)
kde_PDE = ss.gaussian_kde(summed_PDE/CAP*100)
plt.plot(x, kde_PDE(x), label="PDE-EMD-LSTM")

plt.legend()
plt.ylabel("Density")
plt.xlabel("Normalised forecast error (%)")
plt.savefig(f"figures/Error_kde_{edge}.png", dpi = 600)
plt.show()


#%% KDE components
edge = 120
points = 500
x = np.linspace(-edge, edge, points)
HHT.plot_style()

# PDE
eps_PDE = np.load("results/test_error_PDE-EMD-Live-LSTM089.npy")
n,s = np.shape(eps_PDE)
for i in range(s):
    kde_PDE = ss.gaussian_kde(eps_PDE[i])
    plt.plot(x, kde_PDE(x), label=f"k = {i+1}")
plt.legend()
plt.ylabel("Density")
plt.xlabel("Forecast error")
plt.savefig("figures/Error_kde_component_PDE.png", dpi = 600)
plt.show()

# EMD Live
eps_EMD_L = np.load("results/test_error_EMD-Live-LSTM043.npy")
n,s = np.shape(eps_EMD_L)
for i in range(s):
    kde_PDE = ss.gaussian_kde(eps_EMD_L[i])
    plt.plot(x, kde_PDE(x), label=f"k = {i+1}")

plt.legend()
plt.ylabel("Density")
plt.xlabel("Forecast error")
plt.savefig("figures/Error_kde_component_EMD.png", dpi = 600)
plt.show()

#%% Forecast plot
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
