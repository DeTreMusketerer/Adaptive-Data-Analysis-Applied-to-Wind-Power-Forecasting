import numpy as np
import matplotlib.pyplot as plt
import PerformanceMeasures as PM
import PDE_EMD as PDE
import HilbertHuangTransform as HHT

days = 1
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


#%% Histogram
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

#%% Consistency Measure
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

#%% Unification in IMFs
T_line = np.arange(0, n)
unified_4 = np.zeros((n,4,q)).astype(np.float32)
if data_type == "train":
    try:
        unified_4 = np.load(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified_4[i,3,:] = np.sum(decomposition[i,4:,:], axis = 0)
            for j in range(4):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified_4[i,3,:] += decomposition[i,j,:]
                elif j == 2 or j == 3:
                    unified_4[i,2,:] += decomposition[i,j,:]
                else:
                    unified_4[i,j,:] += decomposition[i,j,:]
        np.save(f"Data/PDE_Window_IMFs_q{q}_T{T}_{boundary}.npy", unified_4)
else:
    try:
        unified_4 = np.load(f"Data/PDE_Window_IMFs_Test_q{q}_T{T}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified_4[i,3,:] = np.sum(decomposition[i,4:,:], axis = 0)
            for j in range(s):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified_4[i,3,:] += decomposition[i,j,:]
                elif j == 2 or j == 3:
                    unified_4[i,2,:] += decomposition[i,j,:]
                else:
                    unified_4[i,j,:] += decomposition[i,j,:]
        np.save(f"Data/PDE_Window_IMFs_Test_q{q}_T{T}_{boundary}.npy", unified_4)    

#%% Unification in residual
T_line = np.arange(0, n)
unified = np.zeros((n,s+1,q)).astype(np.float32)
if data_type == "train":
    try:
        unified = np.load(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
    except  FileNotFoundError:
        for i in range(n):
            unified[i,s,:] = np.sum(decomposition[i,s:,:], axis = 0)
            for j in range(s):
                max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                if len(max_pos) + len(min_pos) < 4:
                    unified[i,s,:] += decomposition[i,j,:]
                else:
                    unified[i,j,:] = decomposition[i,j,:]
        np.save(f"Data/PDE_Window_fixed_q{q}_T{T}_s{s}_{boundary}.npy", unified)
else:
     try:
         unified = np.load(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy")
     except  FileNotFoundError:
         for i in range(n):
             unified[i,s,:] = np.sum(decomposition[i,s:,:], axis = 0)
             for j in range(s):
                 max_pos, min_pos, _ = PDE.PDE_EMD.find_extrema(T_line, decomposition[i,j,:])
                 if len(max_pos) + len(min_pos) < 4:
                     unified[i,s,:] += decomposition[i,j,:]
                 else:
                     unified[i,j,:] = decomposition[i,j,:]
         np.save(f"Data/PDE_Window_Test_fixed_q{q}_T{T}_s{s}_{boundary}.npy", unified)

#%% Histogram after unification
component_count_consist = np.zeros(n).astype(np.int32)
for i in range(n):
    for j in range(s+1):
        if not np.allclose(unified[i,j,:],0):
            component_count_consist[i] += 1

HHT.plot_style()
component_count_consist_non_0 = component_count_consist[component_count_consist != 0]
label, count = np.unique(component_count_consist_non_0, return_counts=True)
total = np.sum(count)
plt.bar(label, count/total, width = 0.7)
plt.xlabel("Number of components")
plt.ylabel("Empirical probability")
plt.xticks([3, 4, 5])
plt.savefig(f"figures/Histogram_unified_q{q}_T{T}_s{s}_{boundary}.png", dpi = 600)
plt.show()

green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(component_count_consist_non_0, flierprops=green_diamond)
plt.ylabel("Number of components")
plt.savefig(f"figures/Boxplot_unified_q{q}_T{T}_s{s}_{boundary}.png", dpi = 600)
plt.show()

#%% Consistency Measure
if data_type == "train":
    try:
        CM = np.load(f"Data/CM/CM_q{q}_T{T}_s{s}_{boundary}_unified.npy")
    except FileNotFoundError:
        CM = PM.Consistency_PM(unified, train_mesh)
        np.save(f"Data/CM/CM_q{q}_T{T}_s{s}_{boundary}_unified.npy" ,CM)
else:
    try:
        CM = np.load(f"Data/CM/CM_Test_q{q}_T{T}_s{s}_{boundary}_unified.npy")
    except FileNotFoundError:
        CM = PM.Consistency_PM(unified, test_mesh)
        np.save(f"Data/CM/CM_Test_q{q}_T{T}_s{s}_{boundary}_unified.npy" ,CM)    

HHT.plot_style(13)
spec = plt.pcolormesh(np.arange(1,288), np.arange(1, s+2), CM[:,1:], cmap="viridis_r",
                      shading="auto", vmin=0, vmax=2)
cb = plt.colorbar(spec)
cb.set_label(label='CPM')
plt.gca().invert_yaxis()
plt.ylabel('Component number (k)')
plt.xlabel('Time shift (h)')
plt.savefig(f"figures/CM_PDE_q{q}_T{T}_s{s}_{boundary}_unified.png", dpi=600, bbox_inches="tight")
plt.show()

mean_CM = np.mean(CM, axis=0)
plt.plot(mean_CM[1:175], color="tab:blue")
plt.xlabel("Time shift (h)")
plt.ylabel("$\overline{CPM}_h$")
plt.savefig(f"figures/CM_PDE_Mean_q{q}_T{T}_s{s}_{boundary}_unified.png", dpi=600, bbox_inches="tight")
plt.show()

#%% IO, EEEI
if data_type == "train":
    y_train = np.load("Data/training_data.npy")
    component_count_consist = np.zeros(n).astype(np.int32)
    for i in range(n):
        for j in range(s+1):
            if not np.allclose(unified[i,j,:],0):
                component_count_consist[i] += 1
    n = len(y_train)
    n_actual = len(train_mesh[train_mesh != 0])
    EEEI_array = np.zeros(n_actual).astype(np.float32)
    MIO_array = np.zeros(n_actual).astype(np.float32)
    AIO_array = np.zeros(n_actual).astype(np.float32)
    i = 0
    for t in range(n):
        if train_mesh[t] == 1:
            IMFs = component_count_consist[t]
            unified_small = np.zeros((IMFs,q)).astype(np.float32)
            unified_small[:IMFs-1,:] = unified[t,:IMFs-1,:]
            unified_small[-1,:] = unified[t,-1,:]
            EEEI_array[i] = PM.EEEI(unified_small, y_train[t:t+q])
            IO = PM.Index_orthogonality(unified_small)
            AIO_array[i] = np.mean(IO)
            MIO_array[i] = np.max(IO)
            i += 1
    
    AEEEI = np.mean(EEEI_array)
    AAIO = np.mean(AIO_array)
    AMIO = np.mean(MIO_array)
else:
    y_test = np.load("Data/test_data.npy")
    component_count_consist = np.zeros(n).astype(np.int32)
    for i in range(n):
        for j in range(s+1):
            if not np.allclose(unified[i,j,:],0):
                component_count_consist[i] += 1
    n = len(y_test)
    n_actual = len(test_mesh[test_mesh != 0])
    EEEI_array = np.zeros(n_actual).astype(np.float32)
    MIO_array = np.zeros(n_actual).astype(np.float32)
    AIO_array = np.zeros(n_actual).astype(np.float32)
    i = 0
    for t in range(n):
        if test_mesh[t] == 1:
            IMFs = component_count_consist[t]
            unified_small = np.zeros((IMFs,q)).astype(np.float32)
            unified_small[:IMFs-1,:] = unified[t,:IMFs-1,:]
            unified_small[-1,:] = unified[t,-1,:]
            EEEI_array[i] = PM.EEEI(unified_small, y_test[t:t+q])
            IO = PM.Index_orthogonality(unified_small)
            AIO_array[i] = np.mean(IO)
            MIO_array[i] = np.max(IO)
            i += 1
    
    AEEEI = np.mean(EEEI_array)
    AAIO = np.mean(AIO_array)
    AMIO = np.mean(MIO_array)

#%% IA, IF
IF_matrix = np.zeros((5,3)).astype(np.float32)
IA_matrix = np.zeros((5,3)).astype(np.float32)
f_s = 288
l = 0

for i in range(3,6):
    IF_array = np.zeros(5).astype(np.float32)
    IA_array = np.zeros(5).astype(np.float32)
    unfied_small = unified[component_count_consist == i]
    n = len(unfied_small)
    for t in range(n):
        for j, component in enumerate(unfied_small[t,:,:]):
            IF, IA = HHT.IF_IA(component, f_s, diff = 'taylor', order = 2)
            IF_array[j] += np.mean(IF)
            IA_array[j] += np.mean(IA)
    IF_matrix[:,l] = IF_array/n
    IA_matrix[:,l] = IA_array/n
    l +=1

#%% Grand comparision
T_list = [3,4,5,6,7]
q = 288
s = 4
HHT.plot_style()

legend_list = [f"T = {i}" for i in T_list]

for t in T_list:
    CM = np.load(f"Data/CM/CM_q{q}_T{t}_s{s}_{boundary}_unified.npy")
    mean_CM = np.mean(CM, axis = 0)
    plt.plot(mean_CM[1:175])
    plt.xlabel("Time shift (h)")
    plt.ylabel("$\overline{CPM}_h$")
    
plt.legend(legend_list)
plt.savefig(f"figures/T_Consitency_q{q}_{boundary}.png", dpi=600, bbox_inches="tight")
plt.show()
