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
            1.7: Cleaning up the script adding some docstrings when missing.
                 Removed CEEMDAN and EEMD functionality as well as the support
                 for the CS based decompositions. Also removed plotting
                 functionality and unittests. Added the unification
                 procedure. (19/05/2022)
"""


import numpy as np
import scipy.signal
import PyEMD
import multiprocessing as mp
import time as time


def EMD(y, MAX_ITERATION=1000, max_imf=-1):
    """
    Computes the empirical mode decomposition.

    Parameters
    ----------
    y : ndarray, size=(n,)
        Signal to be decomposed by the EMD.
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs. If this is set to -1 then the method continues
        until the stopping criteria are met. The default is -1.

    Returns
    -------
    IMFs : ndarray, size=(s+1, n)
        Array containing the resulting IMFs and the residual.
    """
    IMFs = PyEMD.EMD(MAX_ITERATION=MAX_ITERATION).emd(y, max_imf=max_imf)
    return IMFs


def get_result(result):
    """
    Multiprocessing callback function.
    """
    global IMFs
    IMFs[result[0],  :np.shape(result[1])[0], :] = result[1]

def live_EMD_MP(t, y, q : int=288, max_imf : int=-1):
    """
    EMD of a window of the signal y of length q.

    Parameters
    ----------
    t : int
        Time index.
    y : ndarray, size=(n,)
        Signal to be decomposed by the EMD.
    q : int, optional
        Window length. The default is 288.
    max_imf : int, optional
        Maximum number of IMFs. The default is -1.

    Returns
    -------
    t : int
        Time index.
    EMD_window : ndarray, size=(s+1, q)
        EMD of y[t:t+q].
    """
    assert (len(y) >= q), "The length of the signal must be longer than the size of the window!"
    if t + q >= len(y):
        assert False, "Not enough previous data for this choice of window length!"
    else:
        print(t)
        EMD_window = EMD(y[t:t+q], MAX_ITERATION=1000, max_imf = max_imf)
        assert np.shape(EMD_window)[0] > 1, \
                    "Unable to decompose signal. Try increasing the window length!"
        return (t, EMD_window)

def decompose_EMD_MP(y, mesh, model_name : str="001", q : int=288, max_imf : int=5):
    """
    Multiprocessing imlpementation of the EMD applied to the wind power data
    used in this thesis. The implies computing the EMD for windows of length
    q at each viable time t.

    Parameters
    ----------
    y : ndarray, size=(n,)
        Signal to be decomposed by the EMD.
    mesh : ndarray, size=(n,)
        Binary array indicating which time points a decomposition can be made
        avoiding to decompose the data at times where data is missing.
    model_name : str, optional
        Name of the decomposition used for saving the result. The default is "001".
    q : int, optional
        Window length. The default is 288.
    max_imf : int, optional
        Maximum number of IMFs. The default is 5.

    Returns
    -------
    IMFs : ndarray, size=(n, max_imf+1, q)
        The EMD for each window of length q of the signal y.
    """
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

    np.save(f"Data/EMD_IMFs_{model_name}.npy", IMFs)
    print((time.time()-t1)/np.sum(mesh))
    return IMFs


def phase(c_a):
    """
    Computes the phase function from the analytic signal y_a.

    Parameters
    ----------
    c_a : ndarray, type=complex, size=(n,)
        Analytic signal.

    Returns
    -------
    theta : ndarray, size=(n,)
        Phase function.
    """
    arg = np.angle(c_a)
    theta = np.unwrap(arg)
    return theta


def IF_IA(c, f_s, diff = 'taylor', order = 2):
    """
    Computes the instantaeous frequency and instantaneous amplitude of a
    signal using the analytical signal approach with a numeric approximation
    of the phase function derivative.

    Taylor based on: Abel, Markus:
    Numerical differentiation: local versus global methods (2005).

    Parameters
    ----------
    c : ndarray, size=(n,)
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
    
    c_a = scipy.signal.hilbert(c)
    n = len(c_a)
    theta = phase(c_a)
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
        omega[2:] = 0.5 * (np.angle(-c_a[2:] * np.conj(c_a[:-2])) + np.pi)
        omega = omega*f_s
    else:
        print("Type of derivative not support. Currently {} are supported.".format(supported_types))
        return
    a = np.absolute(c_a)    
    return omega, a


def HHT(y, f_s, diff = 'trapezoid', order = 2, **kwargs):
    """
    Calculates the Hilbert Huang transform of a signal y.

    Parameters
    ----------
    y : ndarray, size=(n,)
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

    **kwargs
    ----------
    MAX_ITERATION : int, optional
        Maximum number of sifting iterations in EMD and EEMD. The default is 1000.
    max_imf : int, optional
        Maximum number of IMFs in EMD and EEMD. The default is -1.

    Returns
    -------
    omega : ndarray, size=(s, n)
        Instantaneous frequency function.
    a : ndarray, size=(s, n)
        Instantaneous amplitude function.
    """
    MAX_ITERATION = kwargs.get("MAX_ITERATION", "1000")
    max_imf = kwargs.get("max_imf", "-1")
    IMFs = EMD(y, MAX_ITERATION=MAX_ITERATION, max_imf=max_imf)
    J, n = np.shape(IMFs)
    s = J-1
    omega = np.zeros((s, n))
    a = np.zeros((s, n))
    for i in range(s):
        omega[i, :], a[i, :] = IF_IA(IMFs[i], f_s, diff, order)
    return omega, a


def unify_IMFs(IMFs, J, i):
    """
    Unify the components for different windows in terms of what each component
    represents. This consists of a number of steps:
        1) First components above s are summed together to form the
           residual.
        2) Then the residual is checked for number of extrema and energy to make
           sure it is the trend. If it is not the trend the the last IMF is
           joined with the residual to form the new residual.
        4) If the last IMF has less than 3 extrema, then add this to the residual.
        3) Finally, if the number of components after this treatment is less
           than s, then the last component is assumed to be the residual
           and the missing component is assumed to be the last IMFs, i.e.
           the low frequency IMFs.

    Parameters
    ----------
    IMFs : ndarray, size=(max_imf+1, q)
        The components returned by a decomposition method making a maximum of 
        max_imf+1 components. If less than max_imf+1 components were found then
        the empty axes are zeros.
    J : int
        The desired number of components.
    i : int
        Time index.

    Returns
    -------
    IMFs_fixed : ndarray, size=(J, q)
        Fixed version of IMFs with desired size of first axis.
    """
    J_old, q = np.shape(IMFs)
    for k in range(J_old):
        if np.sum(IMFs[k, :]) == 0:
            s_found = k
            break
        elif k == J_old-1:
            s_found = k+1

    if s_found == 0:
        return IMFs[:J, :]
    else:
        IMFs_fixed = np.zeros((J, q), dtype=np.float32)
    
        # Step 1) and 2)
        j = 0
        number_of_extrema = 5
        while number_of_extrema > 4 and np.linalg.norm(IMFs_fixed[J-1, :], ord=2) < 1e-01:
            j += 1
            if s_found > J:
                IMFs_fixed[J-1, :] = np.sum(IMFs[J-j:, :], axis=0)
            else:
                IMFs_fixed[J-1, :] = np.sum(IMFs[s_found-j:, :], axis=0)
    
            d = np.diff(IMFs_fixed[J-1, :])
            d1, d2 = d[:-1], d[1:]
            indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
            indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1
            number_of_extrema = len(indmin) + len(indmax)        

        assert s_found > j, "No IMFs?"

        # Step 3)
        if s_found-j < J-1:
            IMFs_fixed[J-1-(s_found-j):J-1, :] = IMFs[:s_found-j, :]
        elif s_found-j >= J-1:
            IMFs_fixed[:J-1, :] = IMFs[:J-1, :]

        # Step 4)
        d = np.diff(IMFs_fixed[J-2, :])
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
        indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1
        number_of_extrema = len(indmin) + len(indmax)
        if number_of_extrema < 5:
            IMFs_fixed[J-1, :] = np.sum(IMFs_fixed[J-2:, :], axis=0)
            IMFs_fixed[1:J-1, :] = IMFs_fixed[0:J-2, :]
            IMFs_fixed[0, :] = np.zeros(q, dtype=np.float32)
        return IMFs_fixed


def unification_procedure(IMFs, s):
    """
    Parameters
    ----------
    IMFs : ndarray, size=(max_imf+1, q)
        The components returned by a decomposition method making a maximum of 
        max_imf+1 components. If less than max_imf+1 components were found then
        the empty axes are zeros.
    s : int
        The desired number of IMFs.

    Returns
    -------
    IMFs_new_fixed : ndarray, size=(s, q)
        The unified IMFs.
    """
    n, _, q = np.shape(IMFs)
    J = s+1
    IMFs_fixed = np.zeros((n, J, q), dtype=np.float32)
    for i in range(n):
        IMFs_fixed[i, :, :] = unify_IMFs(IMFs, J, i)
    n, J, q = np.shape(IMFs_fixed)
    s = J-1
    s_array = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for k in range(J):
            if np.sum(IMFs_fixed[i, -(k+1), :]) == 0:
                s_array[i] = k
                break
            elif k == s:
                s_array[i] = k+1
    IMFs_new_fixed = np.zeros((n, J, q), dtype=np.float32)
    IMFs_new_fixed[:, -1, :] = IMFs_fixed[:, -1, :]

    for k in range(J):
        if k > 1 and len(s_array == k) != 0:
            IMFs_new_fixed[s_array==k, :k-1, :] = IMFs_fixed[s_array==k, s-(k-1):s, :]
    return IMFs_new_fixed


def main():
    np.random.seed(42)

    # Make a signal
    n = 400
    T = 2
    t = np.linspace(0, T, n)
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
    y = np.sum(c_true, axis=0) + eps

    # EMD
    IMFs = EMD(y)

    # HHT
    IF, IA = HHT(y, f_s)
    return IMFs, IF, IA


if __name__ == "__main__":
    IMFs, IF, IA = main()
