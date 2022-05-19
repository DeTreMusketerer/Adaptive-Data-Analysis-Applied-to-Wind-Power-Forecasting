# -*- coding: utf-8 -*-
"""

Created on Thu Mar 10 15:12:43 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Track changes:
    version 1.0: Hilbert-Huang-Transform (22/02/2022)
			1.1: Bugfixes in the HHT, addition of unittests, color plotting,
                 and more comparison plots. (24/02/2022)
            1.2: Moved from utilities.py to HilbertHuangTransform.py and
                 fixed bugs regarding undefined variables in functions. (10/03/2022)
            1.3: Added keyword arguments to EMD. Added EEMD and CS decomposition
                 methods as inputs to the HHT() function. (17/03/2022)
            1.4: Added plotting functionality to HHT and IF. (21/03/2022)
            1.5: Added CEEMDAN function. Added noise seed to EEMD and CEEMDAN. (06/04/2022)
            1.6: Added liveEMD with multiprocessing together with a wrapper function
                 to decompose data in windows. (27/04/2022)

"""


import numpy as np
import scipy.signal
import PyEMD
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import unittest
import multiprocessing as mp
import time as time
#from NonlinearMatchingPursuit import nonlinear_matching_pursuit


class TestHHT(unittest.TestCase):
    """
    Unittest class.
    Includes test for the phase() function and the IF_IA() function.
    """
    
    def test_phase(self):
        t = np.linspace(0, 1, 500)
        x = np.cos(10*2*np.pi*t)
        x_a = scipy.signal.hilbert(x)
        angle = np.angle(x_a)
        theta = phase(x_a)
        self.assertTrue(np.allclose(angle, ((theta + np.pi) % (2 * np.pi) - np.pi)))

    def test_IF_IA(self):
        t = np.linspace(0, 1, 500)
        x = np.cos(10*2*np.pi*t)
        f_s = 500

        # Forward
        omega_slow, a_slow = IF_IA_slow(x, f_s, diff="forward")
        omega, a = IF_IA(x, f_s, diff="forward")
        self.assertTrue(np.allclose(omega, omega_slow))

        omega_slow, a_slow = IF_IA_slow(x, f_s, diff="taylor", order=3)
        omega, a = IF_IA(x, f_s, diff="taylor", order=3)
        self.assertTrue(np.allclose(omega, omega_slow))

        omega_slow, a_slow = IF_IA_slow(x, f_s, diff="trapezoid")
        omega, a = IF_IA(x, f_s, diff="trapezoid")
        self.assertTrue(np.allclose(omega, omega_slow))


def EMD(x, MAX_ITERATION=1000, max_imf=-1):
    """
    Computes the empirical mode decomposition.

    Parameters
    ----------
    x : ndarray, size=(n,)
        Signal to be decomposed by the EMD.
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs. The default is -1.

    Returns
    -------
    IMFs : ndarray, size=(J, n)
        Array containing the resulting IMFs and the residual.
    """
    IMFs = PyEMD.EMD(MAX_ITERATION=MAX_ITERATION).emd(x, max_imf=max_imf)
    return IMFs


def get_result(result):
    global IMFs
    IMFs[result[0],  :np.shape(result[1])[0], :] = result[1]

def live_EMD_MP(t, signal, q : int=288, max_imf : int=-1):
    assert (len(signal) >= q), "The length of the signal must be longer than the size of the window!"
    if t + q >= len(signal):
        assert False, "Not enough previous data for this choice of window length!"
    else:
        print(t)
        EMD_window = EMD(signal[t:t+q], MAX_ITERATION=1000, max_imf = max_imf)
        assert np.shape(EMD_window)[0] > 1, \
                    "Unable to decompose signal. Try increasing the window length!"
        return (t, EMD_window)

def decompose_EMD_MP(y, mesh, model_name : str="001", q : int=288, max_imf : int=2):
    n = len(y)
    global IMFs
    IMFs = np.zeros((n, max_imf+1, q), dtype=np.float32)
    M = mp.cpu_count()

    t1 = time.time()
    pool = mp.Pool(M)

    for t in range(n):
        if mesh[t] == 1:
            res = pool.apply_async(live_EMD_MP, args=(t, y, q, max_imf), callback=get_result)

    pool.close()
    pool.join()

    np.save(f"Data/EMD_Windows/IMFs_{model_name}.npy", IMFs)
    print((time.time()-t1)/np.sum(mesh))
    return IMFs

def EEMD(x, MAX_ITERATION=1000, max_imf=-1):
    """
    Computes the ensemble empirical mode decomposition.

    Parameters
    ----------
    x : ndarray, size=(n,)
        Signal to be decomposed by the EEMD.
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs. The default is -1.

    Returns
    -------
    IMFs : ndarray, size=(J, n)
        Array containing the resulting IMFs and the residual.
    """
    Class = PyEMD.EEMD(MAX_ITERATION=MAX_ITERATION)
    Class.noise_seed(42)
    IMFs = Class.eemd(x, max_imf=max_imf)
    return IMFs

def CEEMDAN(x, MAX_ITERATION=1000, max_imf=-1):
    """
    Computes the complete ensemble empirical mode decomposition with
    adaptive noise (CEEMDAN).

    Parameters
    ----------
    x : ndarray, size=(n,)
        Signal to be decomposed by the EEMD.
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs. The default is -1.

    Returns
    -------
    IMFs : ndarray, size=(J, n)
        Array containing the resulting IMFs and the residual.
    """
    Class = PyEMD.CEEMDAN(MAX_ITERATION=MAX_ITERATION)
    Class.noise_seed(42)
    IMFs = Class.ceemdan(x, max_imf=max_imf)
    return IMFs

def phase(x_a):
    """
    Computes the phase function from the analytic signal x_a.

    Parameters
    ----------
    x_a : ndarray, type=complex, size=(n,)
        Analytic signal.

    Returns
    -------
    theta : ndarray, size=(n,)
        Phase function.
    """
    arg = np.angle(x_a)
    theta = np.unwrap(arg)
    return theta


def IF_IA_slow(x, f_s, diff = 'taylor', order = 2):
    x_a = scipy.signal.hilbert(x)
    n = len(x_a)
    theta = phase(x_a)
    supported_types = ['forward', 'backward','central','taylor', 'trapezoid']
    omega = np.zeros(n)
    a = np.zeros(n)
    h = 1/f_s
    if diff == 'forward':
        for i in range(n-1):
            omega[i] = (theta[i+1]-theta[i])/h
    elif diff == 'backward':
        for i in range(1,n):
            omega[i] = (theta[i]-theta[i-1])/h
    elif diff == 'central':
        for i in range(1,n-1):
            omega[i] = (theta[i+1]-theta[i-1])/(2*h)
    elif diff == 'taylor':
        for i in range(order,n-order):
            for j in range(1,order+1):
                alpha = 2*(-1)**(j+1)*np.math.factorial(order)**2/(np.math.factorial(order-j)*np.math.factorial(order+j))
                omega[i] += alpha * (theta[i+j]-theta[i-j])/(2*j*h)
    elif diff == 'trapezoid':
        t = np.arange(2, len(x_a))
        omega[2:] = 0.5 * (np.angle(-x_a[t] * np.conj(x_a[t - 2])) + np.pi)
        omega = omega*f_s
    else:
        print("Type of derivative not support. Currently {} are supported.".format(supported_types))
        return
    a = np.absolute(x_a)    
    return omega, a


def IF_IA(x, f_s, diff = 'taylor', order = 2):
    """
    Computes the instantaeous frequency and instantaneous amplitude of a
    signal using the analytical signal approach with a numeric approximation
    of the phase function derivative.

    Taylor based on: Abel, Markus:
    Numerical differentiation: local versus global methods (2005).

    Parameters
    ----------
    x : ndarray, size=(n,)
        Mono-component for which the instantaneous frequency is determined.
    f_s : int
        Sample rate.
    diff : str, optional
        Type of differencing used to estimate differenciation.
        Options are 'forward', 'backward','central','taylor', and 'trapezoid'.
        The default is 'taylor'.
    order : int, optional
        Order of the local differentiation. Only used when diff = 'taylor'.
        The default is 2.

    Returns
    -------
    omega : ndarray, size=(n,)
        Instantaneous frequency function.
    a : ndarray, size=(J, n)
        Instantaneous amplitude function.
    """
    
    x_a = scipy.signal.hilbert(x)
    n = len(x_a)
    theta = phase(x_a)
    supported_types = ['forward', 'backward','central','taylor', 'trapezoid']
    omega = np.zeros(n)
    a = np.zeros(n)
    h = 1/f_s
    if diff == 'forward':
        omega[:-1] = (theta[1:]-theta[:-1])/h
    elif diff == 'backward':
        omega[1:] = (theta[1:]-theta[:-1])/h
    elif diff == 'central':
        omega[1:-1] = (theta[2:]-theta[:-2])/(2*h)
    elif diff == 'taylor':
        for j in range(1,order+1):
            alpha = 2*(-1)**(j+1)*np.math.factorial(order)**2/(np.math.factorial(order-j)*np.math.factorial(order+j))
            if j-order == 0:
                omega[order:-order] += alpha * (theta[order+j:]-theta[order-j:-order-j])/(2*j*h)
            else:
                omega[order:-order] += alpha * (theta[order+j:-order+j]-theta[order-j:-order-j])/(2*j*h)
    elif diff == 'trapezoid':
        omega[2:] = 0.5 * (np.angle(-x_a[2:] * np.conj(x_a[:-2])) + np.pi)
        omega = omega*f_s
    else:
        print("Type of derivative not support. Currently {} are supported.".format(supported_types))
        return
    a = np.absolute(x_a)    
    return omega, a


def HHT(x, f_s, diff = 'trapezoid', order = 2, decomposition_method = "EMD", **kwargs):
    """
    Calculates the Hilbert Huang transform of a signal x.

    Parameters
    ----------
    x : ndarray, size=(n,)
        Signal to be Hilbert Huang transformed.
    f_s : int
        Sample rate of x. Used with decomposition method 'CS' and when
        differencing, however not when diff = 'trapezoid'.
    diff : str, optional
        Type of differencing used to estimate differenciation.
        Options are 'forward', 'backward','central','taylor', and 'trapezoid'.
        The default is 'trapezoid'.
    order : int, optional
        Order of taylor differentiation. Only used when diff = 'taylor'.
        The default is 2.
    decomposition_method : str, optional
        Choice of decomposition method. Options are ['EMD', 'EEMD', 'CS'].
        The default is 'EMD'.

    **kwargs
    ----------
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations in EMD and EEMD. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs in EMD and EEMD. The default is -1.
    Additional options for the nonlinear_matching_pursuit.__init__() call.

    Returns
    -------
    omega : ndarray, size=(J-1, n)
        Instantaneous frequency function.
    a : ndarray, size=(J-1, n)
        Instantaneous amplitude function.
    """
    MAX_ITERATION = kwargs.get("MAX_ITERATION", "1000")
    max_imf = kwargs.get("max_imf", "-1")
    if decomposition_method == "EMD":
        IMFs = EMD(x, MAX_ITERATION=MAX_ITERATION, max_imf=max_imf)
    elif decomposition_method == "EEMD":
        IMFs = EEMD(x, MAX_ITERATION=MAX_ITERATION, max_imf=max_imf)
    elif decomposition_method == "CS":
        C, residual = nonlinear_matching_pursuit(x, f_s, **kwargs).decompose()
        IMFs = np.vstack((C, residual))
    J, n = np.shape(IMFs)
    omega = np.zeros((J-1, n))
    a = np.zeros((J-1, n))
    for i in range(J-1):
        omega[i, :], a[i, :] = IF_IA(IMFs[i], f_s, diff, order)
    return omega, a


def IMF_plot(x, IMFs):
    """
    Plot of the signal x and the IMFs from the EMD.

    Parameters
    ----------
    x : ndarray, size=(n,)
        Observed signal.
    IMFs : ndarray, size=(J, n)
        IMFs from the EMD.
    """
    
    plot_style()

    plt.plot(x)
    plt.ylabel("Signal")
    plt.show()

    J, n = np.shape(IMFs)
    for j in range(J):
        plt.plot(IMFs[j, :])
        if j != J-1:
            plt.ylabel(f"IMF {j+1}")
        elif j == J-1:
            plt.ylabel("Residual")
        plt.show()


def plot_style(fontsize=13):
    """
    Functionality to define plotting specifications. This is useful for
    unifying plotting style.

    Parameters
    ----------
    fontsize : int, optional
        DESCRIPTION. The default is 13.
    """
    
    # Plotting style
    params = {'axes.titlesize': fontsize,
              'axes.labelsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)
    plt.style.use('seaborn-darkgrid')


def HHT_plot(IF, IA, t, f_s, fontsize=13, savename=None, y_upper_lim:float=100):
    """
    Make a color plot of the Hilbert-Huang-Transform computed using HHT().

    Inspired by:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html

    Parameters
    ----------
    IF : ndarray, size=(J-1, n)
        Instantaneous frequency.
    IA : ndarray, size=(J-1, n)
        Instantaneous amplitude.
    t : ndarray, size=(n,)
        Time array.
    f_s : float
        Sample frequency.
    fontsize : int, optional
        Size of font in plot. The default is 13.
    """
    
    Jminus1, n = np.shape(IF)

    # Do not plot beginning and end where IF is 0
    idx = 0
    while IF[0, idx] == 0:
        idx += 1
    begin = idx

    idx = -1
    while IF[0, idx] == 0:
        idx -= 1
    end = idx

    plot_style(fontsize)

    # Plotting
    fig, ax = plt.subplots()
    norm = plt.Normalize(np.min(IA), np.max(IA))
    for j in range(Jminus1):
        lc = colorline(IF[j, begin:end], IA[j, begin:end], t[begin:end], norm=norm, cmap='viridis_r')
    cbar = plt.colorbar(lc)
    cbar.set_label('Amplitude', labelpad=10)
    plt.xlim(t.min(), t.max())
    plt.ylim(-1, y_upper_lim)
    plt.ylabel("Frequency [rad/s]")
    plt.xlabel("Time [s]")
    if savename is not None:
        plt.savefig(savename+".png", dpi=400)
    plt.show()


def colorline(y, z, x, norm, cmap='viridis_r', linewidth=3, alpha=1.0):
    """
    Plot a colored line with coordinates x and y and color amplitude z.
    Optionally specify a colormap, a norm function, and a linewidth.

    Parameters
    ----------
    y : ndarray, size=(n,)
        2nd coordinate.
    z : ndarray, size=(n,)
        3rd coordinate.
    x : ndarray, size=(n,)
        1st coordinate.
    norm : matplotlib norm
        Normalization of color axis.
    cmap : str, optional
        Colormap. The default is 'jet'.
    linewidth : float, optional
        Width of the lines. The default is 3.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    lc : matplotlib LineCollection
        Collection of lines with varying color.
    """
    
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array.

    Parameters
    ----------
    x : ndarray, size=(n,)
        1st coordinate.
    y : ndarray, size=(n,)
        2nd coordinate.

    Returns
    -------
    segments : ndarray, size=(n-1, 2, 2)
        Line segments.
    """
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def components_comparison_plot(t, c_true, c_hat, use_idx=None, savename=None):
    """
    Plot the true components together with the extracted components.

    Parameters
    ----------
    t : ndarray, size=(n,)
        Time array.
    c_true : ndarray, size=(J_true, n)
        The true components.
    c_hat : ndarray, size(J, n)
        The components extracted using an algorithm, e.g. the EMD.
    use_idx : list, len=J_true
        Index of components from c_hat to use for the comparison.
        The default is None.
    savename : str
        Name for saving plot. The default is None.
    """
    plot_style()

    J_true, n = np.shape(c_true)
    J, n = np.shape(c_hat)
    assert (J_true <= J), "It is assumed that the number of extracted components is greater than or equal to the true number of components"

    c_tilde = np.zeros((J_true, n))
    try:
        assert J_true==J
        use_part = False
        for j in range(J_true):
            if j < J_true-1:
                c_tilde[j, :] = c_hat[j, :]
            else:
                c_tilde[j, :] = np.sum(c_hat[j:, :], axis=0)
    except AssertionError:
        use_part = True
        c_tilde[:-1, :] = c_hat[use_idx, :]
        c_tilde[-1, :] = np.sum(c_hat, axis=0)-np.sum(c_hat[use_idx, :], axis=0)

    if use_part is False:
        for j in range(J_true):
            plt.plot(t, c_tilde[j, :], label=f"IMF {j+1}")
            plt.plot(t, c_true[j, :], label=f"True IMF {j+1}")
            plt.legend(loc="upper left")
            plt.xlabel("Time [s]")
            if savename is not None:
                plt.savefig(savename+"_IMF{}.png".format(j+1), dpi=400)
            plt.show()
    elif use_part is True:
        for j, uidx in zip(range(J_true), use_idx+[J_true]):
            plt.plot(t, c_tilde[j, :], label=f"IMF {uidx+1}")
            plt.plot(t, c_true[j, :], label=f"True IMF {uidx+1}")
            plt.legend(loc="upper left")
            plt.xlabel("Time [s]")
            if savename is not None:
                plt.savefig(savename+"_IMF{}.png".format(uidx+1), dpi=400)
            plt.show()


def IF_comparison_plot(t, IF_true, IF_hat, use_idx=None, savename=None):
    """
    Plots the true IF of the components together with the computed IF of the
    extracted components.

    Parameters
    ----------
    t : ndarray, size=(n,)
        Time array.
    IF_true : ndarray, size=(J_true-1, n)
        The true instantaneous frequency.
    IF_hat : ndarray, size=(J-1, n)
        The computed instantaneous frequency.
    """
    
    plot_style()

    Jminus1_true, n = np.shape(IF_true)
    Jminus1, n = np.shape(IF_hat)

    try:
        assert Jminus1_true==Jminus1
        use_part = False
    except AssertionError:
        use_part = True
        IF_hat = IF_hat[use_idx, :]

    # Do not plot beginning and end where IF is 0
    idx = 0
    while IF_hat[0, idx] == 0:
        idx += 1
    begin = idx

    idx = -1
    while IF_hat[0, idx] == 0:
        idx -= 1
    end = idx

    if use_part is False:
        for j in range(Jminus1_true):
            plt.plot(t[begin:end], IF_hat[j, begin:end], label=f"IMF {j+1}")
            plt.plot(t[begin:end], IF_true[j, begin:end], label=f"True IMF {j+1}")
            plt.legend(loc="upper left")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [rad/s]")
            if savename is not None:
                plt.savefig(savename+"_IMF{}.png".format(j+1), dpi=400)
            plt.show()
    elif use_part is True:
        for j, uidx in zip(range(Jminus1_true), use_idx):
            plt.plot(t[begin:end], IF_hat[j, begin:end], label=f"IMF {uidx+1}")
            plt.plot(t[begin:end], IF_true[j, begin:end], label=f"True IMF {uidx+1}")
            plt.legend(loc="upper left")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [rad/s]")
            if savename is not None:
                plt.savefig(savename+"_IMF{}.png".format(uidx+1), dpi=400)
            plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    use = [5] # Options 0, 1, 2, 3, 4, 5

    # Make a signal
    n = 400
    T = 2
    t = np.linspace(0, T, n)
    x = np.zeros(n)
    f_s = n/T

    J_true = 3
    c_true = np.zeros((J_true, n))
    IF_true = np.zeros((J_true-1, n))

    a1 = (0.5 + 0.2*np.cos(2*2*np.pi*t))
    theta1 = np.cos(25*2*np.pi*t)
    c_true[0, :] = a1*theta1
    IF_true[0, :] = 25*np.ones(n)*2*np.pi

    a2 = (0.5 + 0.2*np.cos(0.5*2*np.pi*t))
    theta2 = np.cos((2+np.exp(t*0.7))*2*np.pi*t)
    c_true[1, :] = a2*theta2
    IF_true[1, :] = (2 + np.exp(0.7*t))*2*np.pi + 0.7*np.exp(0.7*t)*2*np.pi*t

    r = t
    c_true[2, :] = r

    eps = np.sqrt(0.0)*np.random.randn(n)

    x = np.sum(c_true, axis=0) + eps

    if 0 in use:
        # Unittests
        unittest.main()
    if 1 in use:
        # Test EMD
        IMFs = EMD(x)
    
        # IMF plot
        IMF_plot(x, IMFs)
    if 2 in use:
        # Test HHT
        IF, IA = HHT(x, f_s, diff='taylor', order=3)
    
        # HHT color plot
        HHT_plot(IF, IA, t, f_s)
    if 3 in use:
        # Plot comparison with true and EMD
        c_hat = EMD(x)
        components_comparison_plot(t, c_true, c_hat)

        # Test HHT
        IF, IA = HHT(x, f_s)
        IF_comparison_plot(t, IF_true, IF)
    if 4 in use:
        c_hat = EEMD(x)
    if 5 in use:
        c_hat = CEEMDAN(x)
        components_comparison_plot(t, c_true, c_hat)
