"""
This script decomposes a signal into IMFs using a PDE based method.
Method imnspired by: 
H. Wang, R. Mann, and E. R. Vrscay, A novel foward-pde approach as
an alternative to empirical mode decomposition, 2018.
and
N. E. Huang et al., The empirical mode decomposition and the hilbert
spectrum for nonlinear and non-stationary time series analysis, 1998
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import time


class PDE_EMD():
    def __init__(self, **kwargs):
        """
        Initiate *PDE* instance.

        Configuration, such as threshold values, can be passed as kwargs.

        Parameters
        ----------
        total_power_thr : float (default 0.005)
            Threshold value on total power per decomposition.
        range_thr : float (default 0.001)
            Threshold for amplitude range (after scaling) per decomposition.
        wang_thr : float (default 0.01)
            Threshold for Wangs stopping criteria.
        MAX_ITERATION : int (default 3000)
            Maximum number of iterations per single "sifting".
        """

        # Declare constants
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.005))
        self.range_thr = float(kwargs.get("range_thr", 0.001))
        self.wang_thr = float(kwargs.get("wang_thr", 0.01))
        self.MAX_ITER = int(kwargs.get("MAX_ITER", 3000))


    def find_extrema(T_line, S):
        """
        Performs extrema detection, where extremum is defined as a point,
        that is above/below its neighbours.
    
        Parameters
        ----------
        T_line : ndarray
            Indicies of the signal.
        S : ndarray
            Signal we want to find the extrema of.
    
        Returns
        -------
        local_max_pos : ndarray
            Array containing the indicies of the local maxima.
        local_min_pos : ndarray
            Array containing the indicies of the local minima.
        indzer : ndarray
            Array containing the indicies of the zero crossings.
    
        """
        
        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]
            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)
    
            indzer = np.sort(np.append(indzer, indz))
    
        # Finds local extrema
        d = np.diff(S)
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
        indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1
    
        # When two or more points have the same value
        if np.any(d == 0):
    
            imax, imin = [], []
    
            bad = d == 0
            dd = np.diff(np.append(np.append(0, bad), 0))
            debs = np.nonzero(dd == 1)[0]
            fins = np.nonzero(dd == -1)[0]
            if debs[0] == 1:
                if len(debs) > 1:
                    debs, fins = debs[1:], fins[1:]
                else:
                    debs, fins = [], []
    
            if len(debs) > 0:
                if fins[-1] == len(S) - 1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []
    
            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k] - 1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(np.round((fins[k] + debs[k]) / 2.0))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(np.round((fins[k] + debs[k]) / 2.0))
    
            if len(imax) > 0:
                indmax = indmax.tolist()
                for x in imax:
                    indmax.append(int(x))
                indmax.sort()
    
            if len(imin) > 0:
                indmin = indmin.tolist()
                for x in imin:
                    indmin.append(int(x))
                indmin.sort()
    
        local_max_pos = T_line[indmax]
        local_min_pos = T_line[indmin]
    
        return local_max_pos, local_min_pos, indzer
    
    
    def end_condition(self, S, IMF):
        """
        Determines if the IMF finding procedure should end
    
        Parameters
        ----------
        S : ndarray
            The signal we started with.
        IMF : ndarray
            Array of all the found IMFs.
            
        Returns
        -------
        bool
            A True or False bool where True ends the IMF finding procedure.
    
        """
        
        tmp = S - np.sum(IMF, axis=0)
        
        if np.max(tmp) - np.min(tmp) < self.range_thr:
            return True
    
        if np.sum(np.abs(tmp)) < self.total_power_thr:
            return True
    
        return False
    
    
    def create_A(n, boundary = "Dirichlet_1"):
        """
        The main functionality for solving the PDE whose solution is the local mean
        of the signal.
        
    
        Parameters
        ----------
        n : int
            Length of the signal you are decomposing.
        boundary : str, optional
            The type of boundary condition. Currently Dirichlet_0, Dirichlet_1, 
            Neumann_0, and Neumann_1 is supported. The default is "Dirichlet_1".
    
        Returns
        -------
        A : ndarray
            the A matrix.
            
        """
        
        supported_boundaries = ["Neumann_0", "Dirichlet_0", "Dirchlet_1", "Neumann_1"]
        diag = np.ones(n) * 2
        offdiag = np.ones(n-1) * -1
        A = np.zeros((n,n)).astype(np.int32)
        A = np.diag(diag, 0) + np.diag(offdiag, -1) + np.diag(offdiag, 1)   
        if boundary == "Neumann_0":
            A[0,1] = -2
            A[n-1, n-2] = -2
        elif boundary == "Neumann_1":
            A[0,:] = A[1,:]
            A[n-1,:] = A[n-2,:]            
        elif boundary == "Dirichlet_0":
            A = A
        elif boundary == "Dirichlet_1":
            A[0,:] = 0
            A[n-1,:] = 0
        else:
            print("boundary type not supported use one of the following {}".format(supported_boundaries))
            return None
        return A
    
    
    def mean_envelope(self, A, signal, spatialline, T, alpha,  t_0 = 0):
        """
        The main functionality for solving the PDE whose solution is the local mean
        of the signal.
        
    
        Parameters
        ----------
        signal : ndarray
            The signal which we want to determine the local mean of.
        spatialline : ndarray
            array of time "samples".
        T : int
            endtime.
        alpha : float
            alpha parameter of the PDE.
        t_0 : int, optional
            starttime. The default is 0
    
        Returns
        -------
        h : ndarray
            the local mean of "signal".
            
        """
        
        n = len(spatialline)
        K = spatialline[-1]
        dx = K/n
        dt = K**2/(4*alpha*n**2)
        N = int(T/dt)
        if N > self.MAX_ITER:
            N = self.MAX_ITER
        const = alpha*dt/(dx**2)
        h = np.zeros(n).astype(np.float32)
        h = copy.copy(signal)
        
        I = np.eye(n)
        for j in range(N):
            h = np.matmul((I - const*A),h)          
        return h
    
    
    def PDE_EMD(self, spatialline, signal, T, max_IMF = 20, plots = False, t_0 = 0,
                boundary = "Dirichlet_1", savefig = False):
        """
        The script for determining the IMFs and residual of a signal. This method
        is inspired by 
    
        Parameters
        ----------
        spatialline : ndarray
            spatial sample points.
        signal : ndarray
            Signal we want to decompose.
        T : int
            End time.
        max_IMF : int, optional
            Maximum amount of IMFs to be found. The default is 20.
        plots : bool, optional
            Determines if the progress should be plotted. The default is False.
        t_0 : int, optional
            Start time. The default is 0.
        boundary : str, optional
            Choses which type of boundary condition is used. Currently, Dirichlet_0,
            Dirichlet_1, Neumann_0 and Neumann_1 are implemented.
            The default is "Dirichlet_1".
        savefig : str, optional
            If True the function saves the generated figures.
            The default is False
        Returns
        -------
        IMFs : ndarray
            Array containing the found IMFs and the residual.
    
        """
        
        n = len(signal)
        IMFNo = 0
        IMF = copy.copy(signal)
        IMFs = np.empty((IMFNo, n))
        r = copy.copy(signal)
        T_line = np.arange(0, n)
        A = PDE_EMD.create_A(n, boundary = boundary)
        while not self.end_condition(signal, IMFs):
            if IMFNo == max_IMF:
                break
            
            max_pos, min_pos, indzer = PDE_EMD.find_extrema(T_line, r)
            extNo = len(max_pos) + len(min_pos)
            if extNo < 2:
                break
            
            if np.shape(min_pos)[0] >= 2:
                min_dist = min(np.diff(min_pos)) * (spatialline[-1]/len(spatialline))
                alpha = 1/((2*np.pi/min_dist)**2)
            else:
                alpha = 1/(4*np.pi**2)
                T = 3

            mean = self.mean_envelope(A, r, spatialline, T, alpha, t_0)
            IMF = r - mean
            
            if np.allclose(IMF,0):
                break
            
            if (np.sqrt(np.mean((mean**2)))/np.sqrt(np.mean((r**2))))**2 < self.wang_thr:
                 break

            IMFNo += 1
            IMFs = np.vstack((IMFs, IMF))
            if plots:
                plt.style.use('seaborn-darkgrid')
                plt.plot(spatialline, r)
                plt.plot(spatialline, mean)
                plt.plot(spatialline, IMF)
                plt.legend(['residual','mean envelope','potential_IMF'])
                if savefig:
                    plt.savefig("figures/IMFprogress_{}_{}.pdf".format(T,IMFNo), dpi = 400)
                plt.show()
                if not np.allclose(IMF,0):
                    plt.plot(IMF)
                    plt.title("IMF {}".format(IMFNo))
                    if savefig:
                        plt.savefig("figures/IMF_{}_{}.pdf".format(T,IMFNo), dpi = 400)
                    plt.show()
            r = r - IMF
        
        if plots:
            plt.plot(r)
            plt.title("Residual")
            if savefig:
                plt.savefig("figures/residual_{}.pdf".format(T), dpi = 400)
            plt.show()
        IMFs = np.vstack((IMFs, r))
        
        return IMFs
    
    
    def PDE_EMD_T(self, spatialline, signal, T, max_IMF, plots = False, t_0 = 0,
                  boundary = "Dirichlet_1", savefig = False):
        """
        Increases the value of T if the chosen value is to small

        Parameters
        ----------
        spatialline : ndarray
            array of the spatial indicies.
        signal : ndarray
            signal to be decomposed.
        T : float
            Final time in.
        max_IMF : int
            maximum amount of IMFs that should be found.
        plots : bool, optional
            Determines if plots are made along the way. The default is False.
        t_0 : int, optional
            Start time. The default is 0.
        boundary : str, optional
            Determines which boundary conditions that should be used.
            The default is "Dirichlet_1".
        savefig : bool, optional
            Determines if the figure should be saved.
            The default is False.

        Returns
        -------
        IMFs : ndarray
            Array containing the found IMFs and the residual.
            
        """

        tmp = T
        k = 1
        while k == 1:
            IMFs = self.PDE_EMD(spatialline, signal, tmp, max_IMF, plots,
                                t_0, boundary, savefig = savefig)
            k = np.shape(IMFs)[0]
            if k == 1:
                tmp +=2
            else: # Hvis du har valgt en fin T værdi burde den komme direkte hertil.
                return IMFs
        return IMFs


if __name__ == '__main__':
    np.random.seed(42)
    samples = 576
    x_0 = 0
    K = 6
    spatialline = np.linspace(x_0,K,samples)

    s1 = 4*np.sin(3*2*np.pi*spatialline)
    s2 = np.sin(2*np.pi*spatialline)
    residual = 3*spatialline
    noise = 0.2*np.random.randn(samples)
    signal = (s1 + s2 + residual + noise)

    # s1 = 0.5*np.cos(2*np.pi*spatialline)
    # s2 = 2*np.cos(0.1*np.pi*spatialline)
    # s3 = 0.8*np.cos(0.5*np.pi*spatialline)
    # signal = (s1 + s2 + s3)
    q = 288
    T = 5
    t_0 = 0
    boundary_type = "Dirichlet_1"
    max_imf = 9

    IMFs = PDE_EMD().PDE_EMD(spatialline, signal, T, max_IMF = 10, plots = True,
                              boundary = boundary_type, savefig = True)

    # Time plot
    # time_list = []
    # component_list = []
    # for i in range(30):
    #     print(i+1)
    #     total_time = 0
    #     time_start = time.time()
    #     IMFs, N_list = PDE_EMD().PDE_EMD(spatialline, signal, i+1,
    #                         max_IMF = 10, plots = True,
    #                         boundary = boundary_type)
    #     time_end = time.time()
    #     total_time += time_end - time_start
    #     component_list.append(len(IMFs))
    #     print(len(IMFs))
    #     time_list.append(total_time)
    # T_line = np.linspace(1,30,30)
    # plt.style.use('seaborn-darkgrid')
    # fig,ax = plt.subplots()
    # ax.plot(T_line, time_list, color = "#1f77b4")
    # ax.set_xlabel("T")
    # ax.set_ylabel("Time [s]")
    # ax2=ax.twinx()
    # ax2.plot(T_line, component_list, color = "#ff7f0e")
    # ax2.set_ylabel("Number of components")
    # fig.legend(["Time", "Components"])
    # plt.show()
    # fig.savefig('figures/T_effect.pdf', dpi=400)
    

    # #Component plots
    # plt.style.use('seaborn-darkgrid')
    # plt.plot(IMFs[-3,:])
    # plt.plot(s1)
    # plt.legend(["IMF", "True component"])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("figures/Monocomponent_extraction_{}_{}.pdf".format(T,"1st"), dpi = 400)
    # plt.show()
    # plt.plot(IMFs[-2,:])
    # plt.plot(s2)
    # plt.legend(["IMF", "True component"])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("figures/Monocomponent_extraction_{}_{}.pdf".format(T,"2nd"), dpi = 400)    
