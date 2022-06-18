# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:13:03 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this module, the indirect performance measures are implemented 
as described in the report
        Adaptive Data Analysis:
        Theoretical Results and an Application to Wind Power Forecasting
            - Section 6.3: Performance Measures of Decompositions

Track changes:
    version 1.0: Index of Orthogonality. (22/02/2022)
            1.1: End effect evaluation index and amplitude error index added.
                 Moved from utilities.py to PerformanceMeasures.py. (10/03/2022)
            1.2: Corrected some notation and updated docstrings.
                 Bugfix in AEI and EEEI. Added Correlation_Coefficient. (16/03/2022)
            1.3: Bugfix in Correlation_Coefficient.
                 Added one_or_two_performance_measure. (21/03/2022)
            1.4: Added consistency performance measure. (26/04/2022)
            1.5: Added sample_entropy. (10/05/2022)
"""


import numpy as np


def Index_orthogonality(c):
    """
    Calculates the index of orthogonality between each pair of mono-components.

    Based on: Huang, Norden E.; Shen, Samuel S P:
    Hilbert-Huang Transform and Its Applications (2014). Page 85

    Parameters
    ----------
    c : ndarray, size=(J, n)
        Array containing the signals which you want to calculate the index
        of orthogonality for.

    Returns
    -------
    IO : ndarray, size=(int(((J-1)**2+J)/2),)
        Array of the pairwise index of orthogonality.
    """
    J = len(c)
    IO = np.zeros(int(((J-1)**2+J)/2))
    i = 0
    for j in range(J):
        for k in range(j+1,J):
            numerator = abs(np.inner(c[j], c[k]))
            denominator = np.linalg.norm(c[j], ord = 2) * np.linalg.norm(c[k], ord = 2)
            IO[i] = numerator/denominator
            i += 1
    return IO


def EEEI(c, x):
    """
    Calculates the end effect evaluation index for a decomposition.

    Based on: A. Hu, X. Yan, and L. Xiang:
    A new wind turbine fault diagnosis method based on ensemble intrinsic
    time-scale decomposition and wpt-fractal dimension (2015).

    Parameters
    ----------
    c : ndarray, size=(J, n)
        Array containing the signals which you want to calculate end effect
        evaluation index for.
    x : ndarray, size=(n,)
        Array of the signal which has been decomposed into c.

    Returns
    -------
    EEEI_eval : float
        End effect evaluation index.
    """
    E_x = np.sqrt(np.mean(x**2))
    E_i = np.sqrt(np.mean(c**2, axis = 1))
    E_c = np.sqrt(np.sum(E_i**2))
    EEEI_eval = abs(E_c-E_x)/E_x
    return EEEI_eval


def AEI(c_true, c_appr, m):
    """
    Calculates the amplitude error index for a decomposition.

    Based on: H. Ding and J. Lv:
    Comparison study of two commonly used methods for envelope fitting of
    empirical mode decomposition (2012).

    Parameters
    ----------
    c_true : ndarray, size=(J, n)
        Array containing the true monocomponents which some signal consists of.
    c_appr : ndarray, size(J, n)
        Array containing the obtained monocomponents resulting from a
        decomposition.
    m : int
        The amount of samples away from the boundary we consider.

    Returns
    -------
    AEI_L : ndarray, size=(J,)
        The amplitude error index for each monocomponent for the left boundary.
    AEI_R : ndarray, size=(J,)
        The amplitude error index for each monocomponent for the right boundary.
    """
    J = np.shape(c_true)[0]
    n = np.shape(c_true)[1]
    AEI_L = np.zeros(J)
    AEI_R = np.zeros(J)
    
    for i in range(J):
        AEI_L[i] = np.mean(abs(c_true[i, :m] - c_appr[i, :m]))
        AEI_R[i] = np.mean(abs(c_true[i, n-m:] - c_appr[i, n-m:]))
    return AEI_L, AEI_R


def Correlation_Coefficient(IMFs):
    """
    Compute the average Pearson correlation coefficient between IMFs.
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Parameters
    ----------
    IMFs : ndarray, size=(s-1, n)
        IMFs.

    Returns
    -------
    Corr : ndarray, size=(int(((s-1)**2+s)/2),)
        Pearson correlation coefficient in the interval [-1, 1].
    """
    s = len(IMFs)
    Corr = np.zeros(int(((s-1)**2+s)/2))
    i = 0
    for j in range(s):
        for k in range(j+1,s):
            join = np.vstack((IMFs[j], IMFs[k]))
            Cov = np.cov(join)
            Corr[i] = Cov[0, 1]/(np.sqrt(Cov[0, 0])*np.sqrt(Cov[1, 1]))
            i += 1
    return Corr


def one_or_two_performance_measure(IMF1, c1, c2):
    """
    The performance measure used in the paper
        Gabriel Rilling; Patrick Flandrin
        One or Two Frequencies? The Empirical Mode Decomposition Answers (2008)

    Parameters
    ----------
    IMF1 : ndarray, size=(n,)
        The high frequency component.
    c1 : ndarray, size=(n,)
        The true high frequency component.
    c2 : ndarray, size=(n,)
        The true low frequency component.

    Returns
    -------
    float
        Performance measure
    """
    return np.sqrt(np.mean(np.abs(IMF1 - c1)**2))/np.sqrt(np.mean(np.abs(c2)**2))


def normalised_squared_error_of_decomposition(IMFs, c1, c2):
    """
    The performance measure used in the paper
        El Hadji S. Diop; Karl Skretting; Abdel-Ouahab Boudraa
        Multicomponent AM–FM signal analysis based on sparse approximation (2019)

    Parameters
    ----------
    IMFs : ndarray, size=(2, n)
        The high frequency component.
    c1 : ndarray, size=(n,)
        The true high frequency component.
    c2 : ndarray, size=(n,)
        The true low frequency component.

    Returns
    -------
    float
        Performance measure
    """
    return np.linalg.norm(IMFs[0, :] - c1)/np.linalg.norm(c1) + np.linalg.norm(IMFs[1, :] - c2)/np.linalg.norm(c2)


def Consistency_PM(IMFs, mesh):
    """
    Performance measure tracking how close decompositions of neighboring windows
    coincide.

    Parameters
    ----------
    IMFs : ndarray, size=(n, s, q)
        The IMFs computed for windows of size q and with a sparsity of s.
    mesh : ndarray, size=(n,)
        Boolean array to keep track of missing data.

    Returns
    -------
    PM : ndarray, size=(s,)
        The consistency performance measure.
    """
    n, s, q = np.shape(IMFs)
    PM = np.zeros((s, q), dtype=np.float32)
    counter = np.zeros(s, dtype=np.int32)
    for i in range(q, n-q):
        if i % 100 == 0:
            print(i)
        if (mesh[i-q : i+q] == np.ones(2*q, dtype=np.int8)).all():
            for j in range(s):
                if np.sum(IMFs[i, j, :]) != 0:
                    counter[j] += 1
                    for shift in range(1, q):
                        denominator = np.linalg.norm(IMFs[i, j, shift:], ord=2)
                        denominator_2 = np.linalg.norm(IMFs[i, j, :-shift], ord=2)
                        if not np.isclose(denominator,0):
                            if not np.isclose(denominator_2,0):
                                PM[j, shift] += np.linalg.norm(IMFs[i, j, shift:] - IMFs[i+shift, j, :-shift], ord=2)/denominator
                                PM[j, shift] += np.linalg.norm(IMFs[i-shift, j, shift:] - IMFs[i, j, :-shift], ord=2)/denominator_2
                        else:
                            PM[j, shift] += 1
    PM = (PM.T/counter).T
    return PM


def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculates the sample entropy of degree m of a time_series.
    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.
    Args:
        time_series: numpy array of time series
        sample_length: length of longest template vector
        tolerance: tolerance (defaults to 0.1 * std(time_series)))
    Returns:
        Array of sample entropies:
            SE[k] is ratio "#templates of length k+1" / "#templates of length k"
            where #templates of length 0" = n*(n - 1) / 2, by definition
    Note:
        The parameter 'sample_length' is equal to m + 1 in Ref[1].
    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    # The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1

    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)

    # Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    # Templates of length 0 matches by definition:
    Ntemp[0] = n * (n - 1) / 2

    for i in range(n - M - 1):
        template = time_series[i:(i + M + 1)]  # We have 'M+1' elements in the template
        rem_time_series = time_series[i + 1:]

        search_list = np.arange(len(rem_time_series) - M, dtype=np.int32)
        for length in range(1, len(template) + 1):
            hit_list = np.abs(rem_time_series[search_list] - template[length - 1]) < tolerance
            Ntemp[length] += np.sum(hit_list)
            search_list = search_list[hit_list] + 1

    sampen = -np.log(Ntemp[1:] / Ntemp[:-1])
    return sampen
