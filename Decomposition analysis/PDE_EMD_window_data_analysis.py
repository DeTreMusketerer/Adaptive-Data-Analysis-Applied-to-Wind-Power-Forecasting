"""
In this script the windows resulting from the PDE-EMD are analysed
"""


import numpy as np
import matplotlib.pyplot as plt
import PerformanceMeasures as PM
import HilbertHuangTransform as HHT


q = 288
T = 6
s = 4
boundary = "Neumann_0"
data_type = "train"
if data_type == "train":
    decomposition = np.load(f"Data/PDE_Window_q{q}_T{T}_{boundary}.npy")
else:
    decomposition = np.load(f"Data/PDE_Window_Test_q{q}_T{T}_{boundary}.npy")
n, m, q = np.shape(decomposition)
train_mesh = np.load(f"Data/train_mesh_q{q}.npy")
test_mesh = np.load(f"Data/test_mesh_q{q}.npy")


# Histogram
component_count = np.zeros(n).astype(np.int32)
for i in range(n):
    for j in range(m):
        if not np.allclose(decomposition[i,j,:],0):
            component_count[i] += 1

component_count = component_count[component_count != 0]
label, count = np.unique(component_count, return_counts=True)
total = np.sum(count)
HHT.plot_style()
plt.bar(label, count/total, width = 0.7)
plt.xlabel("Number of components")
plt.ylabel("Empirical probability")
plt.xticks([3, 4, 5,6,7,8,9,10])
plt.savefig(f"figures/Histogram_q{q}_T{T}_{boundary}.png", dpi = 600)
plt.show()

green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(component_count, flierprops=green_diamond)
plt.ylabel("Number of components")
plt.savefig(f"figures/Boxplot_q{q}_T{T}_{boundary}.png", dpi = 600)
plt.show()

# Consistency Measure
if data_type == "train":
    try:
        CM = np.load(f"Data/CM/CM_q{q}_T{T}_{boundary}.npy")
    except FileNotFoundError:
        CM = PM.Consistency_PM(decomposition, train_mesh)
        np.save(f"Data/CM/CM_q{q}_T{T}_{boundary}.npy" ,CM)
else:
    try:
        CM = np.load(f"Data/CM/CM_Test_q{q}_T{T}_{boundary}.npy")
    except FileNotFoundError:
        CM = PM.Consistency_PM(decomposition, test_mesh)
        np.save(f"Data/CM/CM_Test_q{q}_T{T}_{boundary}.npy" ,CM)


HHT.plot_style()
spec = plt.pcolormesh(np.arange(1,288), np.arange(1, len(CM)+1), CM[:,1:], cmap="viridis_r",
                      shading="auto", vmin=0, vmax=2)
cb = plt.colorbar(spec)
cb.set_label(label='CPM')
plt.gca().invert_yaxis()
plt.ylabel('Component number (k)')
plt.xlabel('Time shift (h)')
plt.savefig(f"figures/CM_PDE_q{q}_T{T}_{boundary}.png", dpi=600, bbox_inches="tight")
plt.show()

mean_CM = np.mean(CM, axis=0)
plt.plot(mean_CM[1:175], color="tab:blue")
plt.xlabel("Time shift (h)")
plt.ylabel("$\overline{CPM}_h$")
plt.savefig(f"figures/CM_PDE_Mean_PM_q{q}_T{T}_{boundary}.png", dpi=600, bbox_inches="tight")
plt.show()


