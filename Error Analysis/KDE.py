"""
In this script the kernel density estimates and for the different methods are
calculated and plotted 
"""


import numpy as np
import Modules.HilberthuangTransform as HHT
import matplotlib.pyplot as plt
import scipy.stats as ss

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


# KDE on components
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