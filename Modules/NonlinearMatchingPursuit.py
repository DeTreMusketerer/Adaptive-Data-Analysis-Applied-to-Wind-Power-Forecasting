# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:22:44 2022

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Track changes:
    version 1.0: Nonlinear Newton type matching pursuit algorithm
                 base implementation. (09/03/2022)
            1.1: Some notation changed according to changes in the report.
                 Added the cvxopt.solver.qp solver option for the BPDN.
                 Also added kwargs in __init__ to adapt the lambda_j
                 sequence. Also tuned default settings. (15/03/2022)
            1.2: Provided show_progress input such that it can be set to
                 False to mute any info displayed during the code. (16/03/2022)
            1.3: Change to the BW_factor variable. Increased resolution in the
                 frequency domain when initialising the frequency. Added a 
                 criterion to ensure a decreasing sequence of initial frequencies. (21/03/2022)
            1.4: Added the \rho parameter to control the over-completeness. (22/03/2022)
            1.5: Altered some default values of kwargs. Made initialising
                 the frequency not crash due to recursion error. (24/03/2022)
            1.6: Reverted the change to the frequency resolution
                 made in v. 1.3. (25/03/2022)
            1.7: Fixed the update of the first entry in the phase function such
                 that it can take values different from 0. (29/03/2022)
            1.8: Added fixed cardinality of V_b functionality. Also added
                 an option to manually give the initial frequencies as input
                 to the algorithm. (06/04/2022)
            1.9: Adaptation according to the implementation in
                 https://github.com/BruntonUWBio/STIMD/blob/master/Decompose_MP_periodic_sym.py
                 - Changed recover_phase() to use trapeziodal rule.
                 - Changed L_theta definition.
                 - Added fast_solve().
                 - Added smooth_theta_conv().
                 - Added lowpass_filter_frequency_response().
                 - Changed update_frequency() to be compatible with the fast
                   algorithm approach.
                 - Added update_frequenct_fast_alg()
                 - Full implementation of fast algorithm.
            1.10: Introduction of base_NMP class. (17/04/2022)
            1.11: Theta0 grid search. Sort components by frequency.
                  Added WindPowerAlg() function. (18/04/2022)
            1.12: Fixed so that lambda_j can be zero without division by zero
                  occuring. Still happens with denominator sometimes? (18/04/2022)
            1.13: Fixing default settings. (26/04/2022)
            1.14: Added a multiprocessing live implementation with a wrapper function
                  to decompose data in windows. (27/04/2022)
            1.15: Enforcing rho greater than or equal to 2 when using adaptive rho. (27/04/2022)
            1.16: Fixed a bug where cvxopt.matrix cannot have dtype float32 so
                  the residual is converted to float64 before using the solver. (28/04/2022)
"""


from math import floor, ceil
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from cvxopt import matrix
from cvxopt.solvers import qp
import doctest
import matplotlib.pyplot as plt
import PyEMD
import multiprocessing as mp
import time as time


class base_NMP(object):
    """
    Base class for nonlinear_matching_pursuit providing additional functionality
    that is less important.
    """
    def loss_function(self, beta, *args):
        """
        Define the loss function for the perturbed linear program for use
        with scipy.optimize.minimize.
        """
        g = args[0]
        Q = args[2]
        loss = np.dot(g, beta) + 1/2 * np.dot(beta, np.dot(Q, beta))
        return loss

    def Jacobian(self, beta, *args):
        """
        Define the Jacobian matrix for the perturbed linear program for use
        with scipy.optimize.minimize.
        """
        g = args[0]
        Q = args[2]
        gradient = g + np.dot(Q, beta)
        return gradient

    def Hessian(self, beta, *args):
        """
        Define the Hessian matrix for the perturbed linear program for use
        with scipy.optimize.minimize.
        """
        Q = args[2]
        Hessian = Q
        return Hessian

    def constraints(self, *args):
        """
        Define constraints for the perturbed linear program for use with
        scipy.optimize.minimize().
        """
        Phi = args[1]
        r = args[3]
        Phi_tilde = np.concatenate((Phi, np.eye(self.n)), axis=1)
        B = np.diag(np.hstack((np.ones(np.shape(Phi)[1]), np.zeros(self.n))))
        equality_constraint = LinearConstraint(Phi_tilde, r, r)
        inequality_constraint = LinearConstraint(B, np.zeros(np.shape(B)[1]), np.full((np.shape(B)[1]), np.inf))
        return equality_constraint, inequality_constraint

    def lowpass_filter_frequency_response(self, k, cutoff):
        """
        Raised cosine low-pass filter with cutoff frequency cutoff.
    
        Parameters
        ----------
        k : ndarray, size=(ne,)
            Frequency modes.
        cutoff : float
            This is the cut-off frequency based on the
            V(\theta, \lambda)-space.
    
        Returns
        -------
        H : ndarray, size=(ne,)
            Frequency response of filter.
        """
        if cutoff != 0:
            H = (-np.cos((k-cutoff)*np.pi/cutoff)+1.)/2.
            b = (np.sign(k)+1.)/2.
            H = b*H+(1.-b)
            c = (np.sign(cutoff-k)+1.)/2.
            H = H*c
        elif cutoff == 0:
            H = np.array([1 if k_mode == 0 else 0 for k_mode in k])
        return H

    def central_difference(self, a):
        """
        Central differencing extending the signal a of length n as
            a_{-1} = 2a_0 - a_1
            a_{n} = 2a_{n-1} - a_{n-2}
        before computing the central difference
            da_j = (a_{j+1} - a_{j-1})/(2*h), for j = 0, ..., n-1
        where h is the sample period.
    
        Parameters
        ----------
        a : ndarray, size=(n,)
            Signal for which the central difference is computed.
    
        Returns
        -------
        da : ndarray, size=(n,)
            Differenced signal.
        """
        n = len(a)
        h = 1. / (n - 1)
        ae = np.hstack(([2*a[0]]-a[1], a, [2*a[-1]-a[-2]]))
        da = (ae[2:]-ae[:-2])/(2*h)
        return da

    def line_search(self, omega_prev, delta_omega):
        """
        Input
        ----------
        omega_prev : ndarray, size=(n,)
            Instantaneous angular frequency.
        delta_omega : ndarray, size=(n,)
            The step direction.

        Returns
        -------
        eta : float
            Maximum step size in the interval [0, 1] ensuring a monotonely
            increasing phase function.
        """
        # Calculate the parameter that makes the frequency positive
        ide = delta_omega > 0
        eta = 1.
        if (np.sum(ide) == 0):
            eta = 1.
        else:
            eta = np.min((np.min(omega_prev[ide] / delta_omega[ide] / 2.), 1.))
        return eta

    def initialise_frequency(self, residual, kappa : float = 2):
        """
        Derive an initial estimate of the wavenumber which the high frequency
        components of the residual/signal are centered around.

        We do this using a simple method where the Fourier transform is
        computed, then a window is slided across the magnitude spectrum,
        then extrema are located and compared to a threshold. Finally, the
        highest frequency extrema exceeding the threshold is selected.
        If no extrema is located then the center frequency in the spectrum is
        used.

        Examples
        ---------
        >>> import numpy as np
        >>> time = np.linspace(0, 1, 400)
        >>> f_s = 399
        >>> residual = np.cos(2*np.pi*5*time) + np.cos(2*np.pi*10*time)
        >>> omega_0 = nonlinear_matching_pursuit(
        ... time, f_s, BW_factor=3, show_progress=False).initialise_frequency(residual, kappa=7)
        Warning: Estimating the initial frequency failed! Trying again with kappa=5.6
        Initial frequency: 62.6748

        Input
        ----------
        residual : ndarray, size=(n,)
            The residual from which a mono-component should be extracted.
        kappa : float
            The number of standard deviations the frequency energy must differ
            from the mean to be considered a peak. The default is 2.

        Returns
        -------
        omega_0 : ndarray, size=(n,)
            Constant array containing the angular frequency that the high
            frequency components of the residual are centered around.
        """
        number_of_freq = self.n//2+1
        residual_hat = np.fft.rfft(residual)
        magnitude = np.abs(residual_hat)
        freq = np.arange(number_of_freq)/(self.h*self.n)
        window_size = int(self.BW_factor*self.n/self.f_s)
        conv = np.convolve(magnitude, np.ones(window_size), mode="same")
        conv_diff = conv[1:] - conv[:-1]
        conv_ext = conv_diff[1:]*conv_diff[:-1]
        extrema = np.where(conv_ext < 0)[0]+1
        value = conv[extrema]
        mean_conv = np.mean(conv)
        std_conv = np.std(conv)

        if self.show_progress is True:
            plt.plot(magnitude)
            plt.show()
            plt.plot(freq*2*np.pi, conv)
            plt.axhline(y=mean_conv+kappa*std_conv, color='r', linestyle='-')
            plt.show()

        if len(extrema) > 0:
            for val, ext in zip(np.flip(value), np.flip(extrema)):
                if val > mean_conv+kappa*std_conv and (freq[ext]*2*np.pi - np.array(self.init_freqs) < -self.BW_factor*2*np.pi).all():
                    center_freq = freq[ext]*2*np.pi
                    break
        else:
            if self.init_freqs[-1]-self.BW_factor*2*np.pi >= 0:
                center_freq = self.init_freqs[-1]-self.BW_factor*2*np.pi
            else:
                center_freq = 0
            print("Warning: Estimating the initial frequency failed!")

        try:
            print(f"Initial frequency: {center_freq:.4f}")
            omega_0 = np.ones(self.n)*center_freq
            self.init_freqs = self.init_freqs + [center_freq]
        except UnboundLocalError:
            if kappa > 0.1:
                print(f"Warning: Estimating the initial frequency failed! Trying again with kappa={kappa*0.8:.1f}")
                omega_0 = self.initialise_frequency(residual, kappa*0.8)
            else:
                if kappa <= 0:
                    print("Warning: Estimating the initial frequency failed! using center_freq=1")
                    center_freq = 1
                    omega_0 = np.ones(self.n)*center_freq
                    self.init_freqs = self.init_freqs + [center_freq]
                    return omega_0
                else:
                    print(f"Warning: Estimating the initial frequency failed! Trying again with kappa={kappa-0.1:.1f}")
                    omega_0 = self.initialise_frequency(residual, kappa-0.1)
        except RecursionError:
            if self.init_freqs[-1]-self.BW_factor*2*np.pi >= 0:
                center_freq = self.init_freqs[-1]-self.BW_factor*2*np.pi
            else:
                center_freq = 0
            print("Warning: Estimating the initial frequency failed!\nInitial frequency: {center_freq:.4f}")
            omega_0 = np.ones(self.n)*center_freq
        return omega_0


class nonlinear_matching_pursuit(base_NMP):
    """
    Implement the non-linear matching pursuit algorithm for decomposing a
    signal into mono-components as in the article
            Thomas Y. Hou and Zuoqiang Shi
            “Data-driven time–frequency analysis” (2013)

    It is assumed that the input signal, f, can be expressed as
        f(t) = \sum_{k=1}^s a_k(t) \cos(\theta_k(t)) + r(t)
    where s is the sparsity, a_k(t)\cos(\theta_k(t)) are mono-components,
    and r(t) is the residual that is O(\epsilon) where \epsilon > 0 is a
    small number.

    The method iteratively solves the optimisation problem

    Minimise     \gamma ||\hat{a}||_1 + ||r(t) - a(t) \cos(\theta(t))||_2^2
    Subject to   a(t)\cos(\theta(t)) \in \mathcal{D}

    where \mathcal{D} = \{a(t)\cos(\theta(t)) : \theta' >= 0, \theta', a \in V(\theta)\}
    and where
    V(\theta) = \{1, \cos(k\theta/(\rho L_{\theta})), \sin(k\theta/(\rho L_{\theta})) : k=1,\dots,floor(\rho \lambda L_{\theta})\}
    is an overcomplete Fourier basis.

    Afterwards, a new residual is defined as
        r(t) = r(t) - a(t) \cos(\theta(t))
    and the algorithm continues until ||r(t)||_2 is less than a threshold \delta.

    The main functionality is found in the functions demodulate() and/or
    decompose(). These functions can run the outer iteration. The inner
    iterations are done in the function MP_step().
    """
    def __init__(self, signal, f_s, delta: float = 5e-02, epsilon : float = 5,
                 lambda_: float = 0.15, gamma: float = 1, rho: float = 2, **kwargs):
        """
        Input
        -------
        signal : ndarray, size=(n,)
            Input multi-component signal.
        f_s : float
            Sample frequency.
        delta : float, optional
            Threshold used as a stopping criteria in the main algorithm.
            The default is 5e-02.
        epsilon : float, optional
            Threshold used in the outer iteration. The default is 5.
        lambda_ : float, optional
            Constant lambda_ <= 0.5. Controls the smoothness of the
            amplitude and frequency functions in comparison with the
            phase function. The default is 0.15.
        gamma : float, optional
            \ell-1 regularisation parameter. The default is 1.
        rho : float, optional
            Over-completeness parameter. The default is 2.

        **kwargs
        ----------
        sparsity : int, optional
            Pre-defined sparsity of the decomposition. The default is None.
        optimizer : str, optional
            Choose the optimizer used for the basis pursuit denoising problem.
            Options are ["scipy", "cvxopt"]. The default is "cvxopt".
        ftol : float, optional
            The threshold for termination used in the scipy
            optimization using SLSQP. The default is 1e-06.
        max_inner_iter : int, optional
            The maximum number of inner iterations before continuing to
            the next lambda_j value or eventually the next component.
            The default is 100.
        xi : float, optional
            Threshold used for numerically stabilising the instantaneous
            frequency update. The default is 0.1.
        kappa : float, optional
            The number of standard deviations the frequency energy must differ
            from the mean to be considered a peak. The default is 3.
        lambda_init : float, optional
            The initial lambda_ value, i.e. lambda_1 in the sequence
            0 < lambda_1 < lambda_2 < ... < lambda_K = lambda_.
            The lambda values are equidistantly spread from lambda_init
            to lambda_ with K values in between. This value should be
            smaller than lambda_. The default is lambda_.
        K : int, optional
            The length of the lambda_j sequence. The default is 1.
        delta_init : float, optional
            The initial threshold in the algorith, i.e. the threshold when
            lambda_1 is used. This should be greater than delta. The used
            thresholds are then equidistantly spread between delta_init and
            delta with K values. The default is delta.
        show_progress : bool, optional
            If True print the progress during optimisation. If False then mute.
            The default is True.
        BW_factor : float, optional
            Window size parameter for the sliding window used for
            initialising the frequency. The window size is defined as
            int(BW_factor*n/f_s). The default is 3.
        use_fast_alg : bool, optional
            If True use the fast algorithm. The default is False.
        adaptive_rho : bool, optional
            If True adapt the over-completeness parameter to have a
            fixed cardinality of V_b. The default is False.
        fixed_V_modes : int, optional
            Used when adaptive_rho is True. The effect is that
            card(V_b) = fixed_V_modes*2 + 1.
            The default is 5.
        auto_init_freq : bool, optional
            If True then automatically initialise the frequency using
            the function defined in this class. If False then initialise
            the frequency for the components with a pre-defined list
            of angular frequencies given in initial_frequencies_list.
            The default is True.
        initial_frequencies_list : list, optional
            The initial frequencies in case auto_init_freq is False.
            The default is list().

        Parameters
        -----------
        n : int
            Number of datapoints.
        h : int
            Sample period.
        init_freqs : list
            List of initial frequencies. Used in initialise_frequency()
            to avoid initialising different components at the same frequency
            range.
        """
        super(nonlinear_matching_pursuit, self).__init__()

        # args
        self.signal = signal
        self.delta = delta
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.gamma = gamma
        self.rho = rho

        # kwargs
        self.sparsity = int(kwargs.get("sparsity", -1))
        self.use_fast_alg = kwargs.get("use_fast_alg", False)
        self.optimizer = kwargs.get("optimizer", "cvxopt")
        self.ftol = kwargs.get("ftol", 1e-06)
        self.max_inner_iter = kwargs.get("max_inner_iter", 5)
        self.xi = kwargs.get("xi", 0.1)
        self.kappa = kwargs.get("kappa", 3)
        self.lambda_init = kwargs.get("lambda_init", 0.001)
        self.K = kwargs.get("K", 20)
        self.delta_init = kwargs.get("delta_init", self.delta)
        self.show_progress = kwargs.get("show_progress", True)
        self.BW_factor = kwargs.get("BW_factor", 3)
        self.adaptive_rho = kwargs.get("adaptive_rho", False)
        self.fixed_V_modes = kwargs.get("fixed_V_modes", 5)
        self.auto_init_freq = kwargs.get("auto_init_freq", True)
        self.initial_frequencies_list = kwargs.get("initial_frequencies_list", list())

        self.supported_optimizer_types = ["scipy", "cvxopt"]

        self.search_size = 70
        self.full_init = False

        # Specify constants
        self.n = len(signal)
        self.f_s = f_s
        self.h = 1/f_s

        # Specify variables
        self.init_freqs = []

    def form_V(self, theta, lambda_j : float = 0.5):
        """
        For the pre-defined \lambda, define the overcomplete Fourier basis.
        Then form the \mathcal{V} matrix.

        Examples
        ---------
        >>> import numpy as np
        >>> n = 2
        >>> theta = np.linspace(0, 2*np.pi, n)
        >>> V = nonlinear_matching_pursuit([0, 1], 1, lambda_=0.5).form_V(theta)
        >>> print(V.astype(np.int8))
        [[ 1  1  0]
         [ 1 -1  0]]

        Input
        ----------
        theta : ndarray, size=(n,)
            Instantaneous phase.
        lambda_j : float, optional
            Smoothness parameter. The default is 0.5.

        Returns
        -------
        V : ndarray, size=(n, card(V(\theta)))
            Matrix containing sampled values from the basis functions
            of the overcomplete Fourier basis.
        """
        L_theta = (theta[-1] - theta[0])/(2*np.pi)
        if self.adaptive_rho is True and lambda_j != 0:
            self.rho = self.fixed_V_modes/(lambda_j*L_theta)
            if self.rho < 2:
                self.rho = 2
            freq_idx_list = np.arange(0, floor(self.rho*lambda_j*L_theta+1e-05)+1)
        elif self.adaptive_rho is True and lambda_j == 0:
            freq_idx_list = np.array([0])
        elif self.adaptive_rho is False:
            freq_idx_list = np.arange(0, floor(self.rho*lambda_j*L_theta)+1)
        assert isinstance(freq_idx_list[0], np.int32) or isinstance(freq_idx_list[0], np.int64), \
            "The frequency index set must be integers. If this is not the case there is a bug in the code!"
        card_V_theta = 2*(len(freq_idx_list)-1) + 1
        V = np.zeros((self.n, card_V_theta))
        for k in freq_idx_list:
            if k == 0:
                V[:, k] = np.ones(self.n)
            else:
                V[:, k*2 - 1] = np.cos(k*theta/(self.rho*L_theta))
                V[:, k*2] = np.sin(k*theta/(self.rho*L_theta))
        return V

    def fast_solve(self, residual, theta, lambda_j):
        """
        The fast algorithm described in the paper
                    Thomas Y. Hou and Zuoqiang Shi
            “Data-driven time–frequency analysis” (2013)
        and based on the python code adaptation in
        https://github.com/BruntonUWBio/STIMD/blob/master/Decompose_MP_periodic_sym.py

        Input
        ----------
        residual : ndarray, size=(n,)
            Residual.
        theta : ndarray, size=(n,)
            Instantaneous phase function.
        lambda_j : float, optional
            Smoothness parameter. The default is 0.5.

        Returns
        -------
        a : ndarray, size=(n,)
            Part of the instantaneous amplitude.
        b : ndarray, size=(n,)
            Part of the instantaneous amplitude. 
        """
        # Interpolate the data to the theta space
        theta_uniform = np.linspace(theta[0], theta[-1], self.n, endpoint=False)
        r_theta = interp1d(theta, residual, kind='cubic')(theta_uniform)

        # Fourier transform
        r_theta_hat = np.fft.fft(r_theta)

        # Generate wave number
        xr_t = theta[-1] - theta[0]

        k_t = np.fft.fftfreq(self.n)*2*np.pi/xr_t*self.n

        # Determine the wavenumber where IMF concentrate around
        L_theta = int(np.round((xr_t)/(2*np.pi)))
        km = 2*np.pi/xr_t*L_theta

        # Extract the IMF from the spectral of the signal
        h_hat = self.lowpass_filter_frequency_response(np.abs(np.abs(k_t)-km), km*lambda_j)
        r_theta_hat_filtered = r_theta_hat * h_hat

        # Translate the spectral of IMF to the original point
        r_theta_hat_filtered_shifted = np.zeros(self.n, dtype=complex)
        if L_theta > 1:
            for j in range(L_theta):
                r_theta_hat_filtered_shifted[self.n-(L_theta-j)] = r_theta_hat_filtered[j]
            for j in range(L_theta, int(np.min((np.ceil(5*L_theta), self.n)))):
                r_theta_hat_filtered_shifted[j - L_theta] = r_theta_hat_filtered[j]
        elif L_theta == 1:
            r_theta_hat_filtered_shifted[0] = r_theta_hat_filtered[1]
        else:
            r_theta_hat_filtered_shifted[0] = r_theta_hat_filtered[0]

        # Compute a_theta(theta) and b_theta(theta)
        env = np.fft.ifft(r_theta_hat_filtered_shifted)
        a_theta = 2*np.real(env)
        b_theta = -2*np.imag(env)

        # Interpolate back to the time space
        a = interp1d(theta_uniform, a_theta, kind='cubic', fill_value='extrapolate')(theta)
        b = interp1d(theta_uniform, b_theta, kind='cubic', fill_value='extrapolate')(theta)

        # What is this exactly??
        #IMF_fit = np.real(np.fft.ifft(r_theta_hat_filtered))
        #IMF = interp1d(theta_uniform, IMF_fit, kind='cubic', fill_value='extrapolate')(theta)
        return a, b

    def delta_omega_lowpass_filter(self, delta_omega, theta, lambda_j):
        """
        Filtering of the instantaneous frequency update. The implementation
        is based on the code in
        https://github.com/BruntonUWBio/STIMD/blob/master/Decompose_MP_periodic_sym.py

        Parameters
        ----------
        delta_omega : ndarray, size=(n,)
            The instantaneous frequency update.
        theta : ndarray, size=(n,)
            The instantaneous phase.
        lambda_j : float
            The smoothness parameter.

        Returns
        -------
        delta_omega_filtered : ndarray, size=(n,)
            The filtered instantaneous frequency.
        """
        delta_omega_extended = np.hstack((delta_omega, np.flip(delta_omega[1:-1])))
        n_extended = len(delta_omega_extended)

        delta_omega_extended_hat = np.fft.fft(delta_omega_extended)
        xr_t = theta[-1] - theta[0]
        L_theta = np.round((xr_t)/(2*np.pi))

        temp = 0
        if (n_extended % 2) == 0:
            temp = n_extended/2
        else:
            temp = (n_extended+1)/2

        k_t = np.hstack((np.arange(0, temp), np.arange(-temp, 0)))*2*np.pi/(2*xr_t) # Like fftfreq
        km = 2*np.pi/xr_t * L_theta
        h_hat = self.lowpass_filter_frequency_response(np.abs(k_t), km*lambda_j)
        delta_omega_extended_hat_filtered = delta_omega_extended_hat * h_hat

        delta_omega_extended_filtered = np.real(np.fft.ifft(delta_omega_extended_hat_filtered))
        delta_omega_filtered = delta_omega_extended_filtered[:self.n]
        return delta_omega_filtered

    def perturbed_LP_setup(self, residual, theta, lambda_j : float = 0.5):
        """
        Setting up the perturbed linear program (LP) which is equivalent to
        BPDN.

        Examples
        ---------
        >>> import numpy as np
        >>> n = 2
        >>> residual = np.array([0, 2])
        >>> theta = np.linspace(0, 2*np.pi, n)
        >>> h, Phi, Q, r, V = nonlinear_matching_pursuit(
        ... [0, 1], 1, lambda_=0.5, gamma=1.0).perturbed_LP_setup(residual, theta)
        >>> print(Phi.astype(np.int8))
        [[ 1  1  0  0  0  0 -1 -1  0  0  0  0]
         [ 1 -1  0  0  0  0 -1  1  0  0  0  0]]
        >>> print(h.astype(np.int8))
        [1 1 1 1 1 1 1 1 1 1 1 1 0 0]
        >>> print([int(Q[i, i]) for i in range(4*3+2)])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        >>> print(int(np.sum(Q)))
        2

        Input
        ----------
        residual : ndarray, size=(n,)
            The current residual as given in the outer iteration of the
            algorithm.
        theta : ndarray, size=(n,)
            The instantaneous phase.
        lambda_j : float, optional
            Smoothness parameter. The default is 0.5.

        Returns
        -------
        g : ndarray, size=(4*card(V(\theta))+n,)
            Optimisation problem parameters in the \ell_1 part.
        Phi : ndarray, size=(n, 4*card(V(\theta)))
            Constraint matrix for the \ell_1 part.
        Q : ndarray, size=(4*card(V(\theta))+n)
            Diagonal matrix for the quadratic part of the program.
        r : ndarray, size=(n)
            The residual vector in the equality constraint.
        V : ndarray, size=(n, card(V(\theta)))
            Matrix containing sampled values from the basis functions
            of the overcomplete Fourier basis.
        """
        V = self.form_V(theta, lambda_j=lambda_j)
        A = np.concatenate((np.dot(np.diag(np.cos(theta)), V), np.dot(np.diag(np.sin(theta)), V)), axis=1)
        Phi = np.concatenate((A, -A), axis=1)
        four_card_V = np.shape(Phi)[1]
        r = residual
        g = np.hstack((self.gamma*np.ones(four_card_V), np.zeros(self.n)))
        Q = np.diag(np.hstack((np.zeros(four_card_V), np.ones(self.n))))
        return g, Phi, Q, r, V

    def solve_BPDN(self, residual, theta, lambda_j : float = 0.5):
        """
        Solve the \ell-1 regularised least square problem
        a.k.a. basis pursuit denoising (BPDN).

        Different methods are available:
            "scipy":
                - The parameters are initialised randomly.
                - The optimization problem is solved using
                  scipy.optimize.minimize with the SLSQP method.
            "cvxopt":
                - The optimisation problem is solved using
                  cvxopt.solver.qp.

        Input
        ----------
        residual : ndarray, size=(n,)
            The current residual as given in the outer iteration of the
            algorithm.
        theta : ndarray, size=(n,)
            The instantaneous phase.
        lambda_j : float, optional
            Smoothness parameter. The default is 0.5.

        Returns
        -------
        a : ndarray, size=(n,)
            Part of the instantaneous amplitude.
        b : ndarray, size=(n,)
            Part of the instantaneous amplitude. 
        """
        if self.optimizer == "scipy":
            h, Phi, Q, r, V = self.perturbed_LP_setup(residual, theta, lambda_j=lambda_j)
            four_card_V = np.shape(Phi)[1]
            zeta_0 = np.sqrt(0.5)*np.abs(np.random.randn(four_card_V+self.n))
            equality_constraint, inequality_constraint = self.constraints(h, Phi, Q, r)
            result = minimize(self.loss_function, zeta_0, args=(h, Phi, Q, r),
                              jac=self.Jacobian, method="SLSQP",
                              constraints=(equality_constraint, inequality_constraint),
                              options={"ftol": self.ftol})
            beta = result.x
            x = beta[:int(four_card_V/2)] - beta[int(four_card_V/2): four_card_V]
            a_hat = x[:int(four_card_V/4)]
            b_hat = x[int(four_card_V/4):]
        elif self.optimizer == "cvxopt":
            g, Phi, Q, r, V = self.perturbed_LP_setup(residual, theta, lambda_j=lambda_j)

            four_card_V = int(4*np.shape(V)[1])
            Phi_tilde = matrix(np.concatenate((Phi, np.eye(self.n)), axis=1))
            B = matrix(np.concatenate((np.diag(-np.ones(four_card_V)), np.zeros((four_card_V, self.n))), axis=1))
            zeros = matrix(0.0, (four_card_V, 1))
            g = matrix(np.expand_dims(g, -1))
            r = matrix(np.expand_dims(r.astype(np.float64), -1))
            beta = np.array(qp(matrix(Q), g, B, zeros, Phi_tilde, r,
                               options={'show_progress': self.show_progress})["x"])[:, 0]
            x = beta[:int(four_card_V/2)] - beta[int(four_card_V/2): four_card_V]
            a_hat = x[:int(four_card_V/4)]
            b_hat = x[int(four_card_V/4):]
        else:
            assert self.optimizer in self.supported_optimizer_types, \
                f"The BPDN optimizer should be one of {self.supported_optimizer_types}"
        a = np.dot(V, a_hat)
        b = np.dot(V, b_hat)
        return a, b

    def update_frequency(self, omega_prev, theta_prev, a, b, lambda_j : float = 0.5,
                         do_filtering : bool = True):
        """
        Update the frequency by exploiting the trigonometric identity
            a\cos(\theta) + b\sin(\theta) = A\cos(\theta+\phi)
        where A = \sqrt{a^2 + b^2} and \phi = \arctan(b/a).

        A step size \eta is determined using line search in order to ensure a
        positive instantaneous angular frequency.

        The step direction is called delta_theta and is computed as
            \Delta\theta = (a * b' - b a')/(a^2 + b^2)
        where the derivatives are computed using a 1st order backward
        differencing scheme and fixing b'[0] = b'[1], a'[0] = b'[1].

        Examples
        ---------
        >>> import numpy as np
        >>> n = 3
        >>> f_s = 1/2
        >>> omega_prev = np.array([np.pi/2, np.pi/2, np.pi/2])
        >>> theta_prev = np.array([0, np.pi, 2*np.pi])
        >>> a = np.cos(theta_prev)
        >>> b = np.cos(theta_prev/2)
        >>> omega = nonlinear_matching_pursuit(
        ... [0, 1, 2], f_s, show_progress=False).update_frequency(omega_prev, np.zeros(n), a, b, do_filtering=False)
        >>> print(omega.astype(np.float16))
        [1.178 0.785 1.178]

        >>> n = 4
        >>> f_s = 1/2
        >>> omega_prev = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
        >>> a = np.array([1.0, 0.01, 0.01, 0.5])
        >>> b = np.array([2.0, 0.01, 0.01, 1])
        >>> omega = nonlinear_matching_pursuit(
        ... [0, 1, 2, 3], f_s).update_frequency(omega_prev, np.zeros(n), a, b, do_filtering=False)
        >>> print(omega.astype(np.float16))
        [1.577 1.57  1.564 1.559]

        Input
        ----------
        omega_prev : ndarray, size=(n,)
            Instantaneous angular frequency.
        a : ndarray, size=(n,)
            Part of the instantaneous amplitude.
        b : ndarray, size=(n,)
            Part of the instantaneous amplitude.
        lambda_j : float, optional
            Smoothness parameter. The default is 0.5.
        do_filtering : bool, optional
            Boolean input to decide whether or not to use the low-pass
            filtering. This should be set to true in applications.
            The default is True.

        Returns
        -------
        omega_new : ndarray, size=(n,)
            Updated instantaneous angular frequency.
        """
        # Compute the step direction
        a_diff = self.central_difference(a)
        b_diff = self.central_difference(b)
        denominator = a**2+b**2
        denominator[denominator < np.max(denominator)/100] = np.max(denominator)/100 # Avoid division by zero
        delta_omega = (a*b_diff-b*a_diff)/denominator

        # Numerical stability (can this be done faster?)
        check_threshold = np.where(denominator<self.xi)[0]
        check_threshold_above = np.where(denominator>=self.xi)[0]
        # assert len(check_threshold_above) >=2, "Too many unstable denominators!"
        if len(check_threshold) > 0 and len(check_threshold_above) >= 2:
            for idx_critical in check_threshold:
                idx_below = check_threshold_above[check_threshold_above < idx_critical]
                idx_above = check_threshold_above[check_threshold_above > idx_critical]
                if len(idx_below) == 0:
                    delta_omega[idx_critical] = delta_omega[idx_above[0]]
                elif len(idx_above) == 0:
                    delta_omega[idx_critical] = delta_omega[idx_below[-1]]
                else:
                    slope = (delta_omega[idx_above[0]]-delta_omega[idx_below[-1]])/(idx_above[0] - idx_below[-1])
                    intercept = delta_omega[idx_below[-1]]
                    delta_omega[idx_critical] = intercept + slope*(idx_critical-idx_below[-1])
        else:
            if self.show_progress is True:
                print("No correction of frequency update due to numeric instability.")

        if do_filtering is True:
            delta_omega = self.delta_omega_lowpass_filter(delta_omega, theta_prev, lambda_j=lambda_j)

        # Compute the step
        eta = self.line_search(omega_prev, delta_omega)
        omega_new = omega_prev - eta*delta_omega
        return omega_new

    def recover_phase(self, omega, residual=None, env=None, theta_prev0:float=0, a0: float=1, b0: float=0):
        """
        Recover the instantaneous phase from the instantaneous frequency by
        integration (summation), assuming constant frequency within the time
        between two consecutive samples. This function uses the composite
        trapezoidal rule for numerical integration, i.e.
            \theta[0] = theta_prev0 - arctan(b0/a0)
            \theta[1] = theta[0] + (\omega[0] + omega[1])/2 * h
            \theta[2] = theta[1] + (\omega[1] + omega[2])/2 * h
            ...

        Examples
        ---------
        >>> import numpy as np
        >>> n = 3
        >>> theta = nonlinear_matching_pursuit(
        ... [0, 1, 2], 1/2).recover_phase(np.array([5, 8, 6]))
        >>> print(theta.astype(np.int8))
        [ 0 13 27]

        Parameters
        ----------
        omega : ndarray, size=(n,)
            Instantaneous angular frequency.
        theta_prev0 : float
            Previous phase function in the first entry, i.e. \theta[0].
            The default is 0.
        a : float
            Part of the amplitude. Used to update the first entry in the
            phase function. The default is 1.
        b : float
            Part of the amplitude. Used to update the first entry in the
            phase function. The default is 0.

        Returns
        -------
        theta : ndarray, size=(n,)
            Instantaneous phase.
        """
        theta = np.zeros(self.n)
        cumt = cumtrapz(omega, dx = self.h)
        if self.use_fast_alg is True and residual is not None and env is not None:
            theta[0] = self.theta0_grid_search(residual, cumt, env, search_size=self.search_size)
        if self.use_fast_alg is False and residual is not None and env is not None:
            theta[0] = self.theta0_grid_search(residual, cumt, env, search_size=self.search_size)
            #theta[0] = theta_prev0 - np.arctan(b0/a0) # This is a problem for fast algorithm.        
        theta[1:] = theta[0] + cumt
        return theta

    def theta0_grid_search(self, residual, cum_omega, env, search_size):
        """
        Finding the first entry in the phase function when integrating the
        instantaneous frequency to the instantaneous phase in the
        recover_phase() function.
        """
        error = np.zeros(search_size)
        search_list = np.linspace(0, 2*np.pi, search_size, endpoint=False)
        for i, theta0 in enumerate(search_list):
            theta = np.zeros(self.n)
            theta[0] = theta0
            theta[1:] = theta[0] + cum_omega
            error[i] = np.linalg.norm(residual-env*np.cos(theta), ord=2)
        best_theta0 = search_list[np.argmin(error)]
        return best_theta0

    def update_frequency_fast_alg(self, omega_prev, theta_prev, a, b, lambda_j: float=0.5):
        """
        The frequency update when using the fast algorithm. The implementation
        is based on
        https://github.com/BruntonUWBio/STIMD/blob/master/Decompose_MP_periodic_sym.py

        Parameters
        ----------
        omega_prev : ndarray, size=(n,)
            Previous instantaneous frequency.
        theta_prev : ndarray, size=(n,)
            Previous instantaneous phase.
        a : ndarray, size=(n,)
            The real part of the envelope.
        b : ndarray, size=(n,)
            The imagniary part of the envelope.
        lambda_j : float, optional
            The smoothness parameter. The default is 0.5.

        Returns
        -------
        omega_new : ndarray, size=(n,)
            The new instantaneous frequency.
        """
        # Compute the step direction
        a_diff = self.central_difference(a)
        b_diff = self.central_difference(b)
        denominator = a**2+b**2
        denominator[denominator < np.max(denominator)/100] = np.max(denominator)/100 # Avoid division by zero
        delta_omega = (a*b_diff-b*a_diff)/denominator
        delta_omega_filtered = self.delta_omega_lowpass_filter(delta_omega, theta_prev, lambda_j)

        eta = self.line_search(omega_prev, delta_omega_filtered)

        # update the frequency
        # L_theta = np.round((theta_prev[-1]-theta_prev[0])/(2*np.pi))
        # omega_new = 2*np.pi*L_theta/(theta_prev[-1]-theta_prev[0])*omega_prev - eta*delta_omega_filtered
        omega_new = omega_prev - eta*delta_omega_filtered
        return omega_new

    def MP_step(self, residual):
        """
        Find the component A\cos(\theta) which best matches the residual
        using the non-linear matching pursuit algorithm.

        Parameters
        ----------
        residual : ndarray, size=(n,)
            The current residual from which a component is extracted
            in the matching pursuit algorithm.

        Returns
        -------
        A : ndarray, size=(n,)
            Instantaneous amplitude.
        theta : ndarray, size=(n,)
            Instantaneous phase.
        omega : ndarray, size=(n,)
            Instantaneous angular frequency.
        """
        # Initialise variables
        a = np.zeros(self.n)
        b = np.zeros(self.n)
        A = np.zeros(self.n)
        if self.full_init is True:
            omega_0 = self.omega0[self.k, :]
        elif self.auto_init_freq is True:
            omega_0 = self.initialise_frequency(residual, kappa=self.kappa)
        elif self.auto_init_freq is False:
            omega_0 = np.ones(self.n)*self.initial_frequencies_list[self.k]
        omega_prev = np.zeros(self.n)
        omega_new = np.copy(omega_0)
        theta_prev = np.ones(self.n)*-1
        theta_new = self.recover_phase(omega_0)

        # Create sequence of V-spaces
        for lambda_j, delta_j in zip(np.linspace(self.lambda_init, self.lambda_, self.K), np.flip(np.linspace(self.delta, self.delta_init, self.K))):
            theta_prev = np.ones(self.n)*-1
            inner_iter = 0
            # Run the inner iterations
            while np.linalg.norm(theta_new-theta_prev, ord=2) > delta_j:
                omega_prev = np.copy(omega_new)
                theta_prev = np.copy(theta_new)

                if self.use_fast_alg is False:
                    # Solve BPDN
                    a, b = self.solve_BPDN(residual, theta_prev, lambda_j)
                    # Update frequency
                    omega_new = self.update_frequency(omega_prev, theta_prev,
                                                      a, b, lambda_j=lambda_j)
                elif self.use_fast_alg is True:
                    # Solve optimisation problem
                    a, b = self.fast_solve(residual, theta_prev, lambda_j)
                    # Update frequency
                    omega_new = self.update_frequency_fast_alg(omega_prev, theta_prev,
                                                               a, b, lambda_j=lambda_j)

                theta_new = self.recover_phase(omega_new, residual, np.sqrt(a**2 + b**2), theta_prev[0], a[0], b[0])
                if self.show_progress is True:
                    print(f"Iteration: {self.k}\t lambda_j: {lambda_j:.2f}\t Evaluate norm: {np.linalg.norm(theta_new-theta_prev, ord=2):.3f}")
                inner_iter += 1
                if inner_iter == self.max_inner_iter:
                    if lambda_j < self.lambda_:
                        if self.show_progress is True:
                            print("Maximum inner iterations reached! Continuing to the next lambda_j value.")
                    else:
                        if self.show_progress is True:
                            print("Maximum inner iterations reached! Giving up on this component.")
                    break

        # Return mode
        A = np.sqrt(a**2 + b**2)
        theta = np.copy(theta_new)
        omega = np.copy(omega_new)
        return A, theta, omega

    def demodulate(self):
        """
        Demodulates a multi-component signal.

        Returns
        -------
        instantaneous_amplitude : ndarray, size=(s, n)
            Instantaneous amplitude \{a_k(t)\}_{k=1}^s.
        instantaneous_phase : ndarray, size(s, n)
            Instantaneous phase \{\theta_k(t)\}_{k=1}^s.
        instantaneous_frequency : ndarray, size(s, n)
            Instantaneous frequency \{\omega_k(t)\}_{k=1}^s.
        residual : ndarray, size(n,)
            The residual signal.
        """
        # Declare variables
        if self.sparsity != -1:
            instantaneous_amplitude = np.zeros((self.sparsity, self.n))
            instantaneous_phase = np.zeros((self.sparsity, self.n))
            instantaneous_frequency = np.zeros((self.sparsity, self.n))
        else:
            instantaneous_amplitude = np.zeros((0, self.n))
            instantaneous_phase = np.zeros((0, self.n))
            instantaneous_frequency = np.zeros((0, self.n))
        residual = self.signal

        if self.sparsity != -1:
            for k in range(self.sparsity):
                if self.show_progress is True:
                    print("")
                self.k = k
                a, theta, omega = self.MP_step(residual)
                instantaneous_amplitude[k, :] = a
                instantaneous_phase[k, :] = theta
                instantaneous_frequency[k, :] = omega
                residual = self.signal - np.sum(instantaneous_amplitude*np.cos(instantaneous_phase), axis=0)
        else:
            k = 0
            while np.linalg.norm(residual, ord=2) > self.epsilon:
                if self.show_progress is True:
                    print("")
                self.k = k
                a, theta, omega = self.MP_step(residual)
                instantaneous_amplitude = np.concatenate((instantaneous_amplitude, np.expand_dims(a, axis=0)), axis=0)
                instantaneous_phase = np.concatenate((instantaneous_phase, np.expand_dims(theta, axis=0)), axis=0)
                instantaneous_frequency = np.concatenate((instantaneous_frequency, np.expand_dims(omega, axis=0)), axis=0)
                residual = self.signal - np.sum(instantaneous_amplitude*np.cos(instantaneous_phase), axis=0)
                k += 1

        # Sort components according to frequency (high to low)
        mean_IF = np.mean(instantaneous_frequency, axis=1)
        sort = np.flip(np.argsort(mean_IF))
        return instantaneous_amplitude[sort, :], instantaneous_phase[sort, :], instantaneous_frequency[sort, :], residual

    def decompose(self):
        """
        Decomposes a multi-component signal.

        Returns
        -------
        C : ndarray, size=(s, n)
            Matrix containing the mono-components of the signal in the rows.
        residual : ndarray, size=(n,)
            The residual signal.
        """
        instantaneous_amplitude, instantaneous_phase, _, residual = self.demodulate()
        C = instantaneous_amplitude*np.cos(instantaneous_phase)
        return C, residual


def get_result(result):
    global IMFs
    IMFs[result[0],  :, :] = result[1]
    if result[0] % 100 == 0:
        np.save("Data/CS_Windows/IMFs_temp.npy", IMFs[:result[0], :, :])

def live_NMP_MP(t, signal, EMD_residual, initial_frequencies_list : list=list(), use_fast_alg : bool=True,
                q : int=288, s : int=4, lambda_K : float=0.5):
    assert (len(signal) >= q), "The length of the signal must be longer than the size of the window!"
    if t + q >= len(signal):
        assert False, "Not enough previous data for this choice of window length!"
    else:
        print(t, initial_frequencies_list)
        IMFs_window, residual_window = nonlinear_matching_pursuit(signal[t:t+q]-EMD_residual, q, sparsity=s,
                                                    max_inner_iter=1, lambda_init=0.001, lambda_=lambda_K,
                                                    K=20, show_progress=False, adaptive_rho=True,
                                                    use_fast_alg=use_fast_alg, initial_frequencies_list=initial_frequencies_list,
                                                    auto_init_freq=False).decompose()
        CS_window = np.vstack((IMFs_window, residual_window, EMD_residual))
        assert np.shape(CS_window)[0] > 1, \
                    "Unable to decompose signal. Try increasing the window length!"
        return (t, CS_window)


def decompose_NMP_MP(y, mesh, model_name : str="001", s : int=4, q : int=288,
                      use_fast_alg : bool=True, lambda_K : float=0.5, data_ = "training"):
    n = len(y)
    EMD = np.load(f"Data/EMD_Window_q{q}_{data_}_data.npy", mmap_mode="r")
    EMD_IF = np.load(f"Data/EMD_Window_q{q}_{data_}_data_IF.npy", mmap_mode="r")

    global IMFs
    IMFs = np.zeros((n, s+2, q), dtype=np.float32)
    M = mp.cpu_count()

    t1 = time.time()
    pool = mp.Pool(M)

    for t in range(n):
        if mesh[t] == 1:
            freq = np.zeros(s, dtype=np.float32)
            for k, arg in enumerate(range(s)):
                freq[k] = np.mean(EMD_IF[t, arg, 2:-2])
            sort = np.flip(np.argsort(freq))
            initial_frequencies_list = list(freq[sort])
            if initial_frequencies_list[-1] == 0.0:
                initial_frequencies_list[-1] = 20.84

            res = pool.apply_async(live_NMP_MP, args=(t, y, EMD[t, -1, :], initial_frequencies_list,
                                                      use_fast_alg, q, s, lambda_K),
                             callback=get_result)             

    pool.close()
    pool.join()

    np.save(f"Data/CS_Windows/IMFs_{model_name}_{data_}.npy", IMFs)
    with open(f"Data/CS_Windows/Settings_{model_name}.txt", "w") as file:
        file.write(f"Algorithm settings for CS decomposition number {model_name}\n\n")
        file.write(f"Window length: {q}\n")
        file.write(f"Use fast algorithm: {use_fast_alg}\n")
        file.write(f"Sparsity: s+2={s+2}\n")
        file.write(f"lambda_K: {lambda_K}\n\n")
        file.write("lambda_init=0.01, K=20, max_inner_iter=1, V_modes=5\n")
        file.write(f"Average time pr. decomposition {(time.time()-t1)/np.sum(mesh):.5e} with {M} CPUs")
    return IMFs


if __name__ == '__main__':
    np.random.seed(42)
    use = [1] # Options 0, 1, 3

    # Make a signal
    n = 150
    T = 0.65
    t = np.linspace(0, T, n)
    x = np.zeros(n)
    f_s = n/T

    J_true = 2
    c_true = np.zeros((J_true+1, n))
    IF_true = np.zeros((J_true+1, n))

    a1 = 1
    theta1 = 10*2*np.pi*t
    c_true[0, :] = a1*np.cos(theta1)
    IF_true[0, :] = 10*2*np.pi*np.ones(n)

    a2 = 0.5 #(0.5 + 0.2*np.cos(np.pi*t))
    theta2 = 4*2*np.pi*t
    c_true[1, :] = a2*np.cos(theta2+1)
    IF_true[1, :] = 4*np.ones(n)*2*np.pi

    # a3 = (0.5 + 0.2*np.cos(0.5*2*np.pi*t))
    # theta3 = (2+0.5*np.cos(t*0.05*2*np.pi))*2*np.pi*t
    # c_true[2, :] = a3*np.cos(theta3)
    # IF_true[2, :] = (2 + 0.5*np.cos(t*0.05*2*np.pi))*2*np.pi - 0.05*2*np.pi*np.sin(t*0.05*2*np.pi)*2*np.pi*t

    r = 10 + 10*t**2
    c_true[2, :] = r

    eps = np.sqrt(0.05)*np.random.randn(n)

    x = np.sum(c_true, axis=0) + eps

    if 0 in use:
        # Doctests
        doctest.testmod()
    if 1 in use:
        # C, residual = nonlinear_matching_pursuit(x, f_s, sparsity=J_true, lambda_init = 0.001, lambda_ = 0.2, K = 20,
        #                                          show_progress=False, max_inner_iter=1, auto_init_freq=False,
        #                                          initial_frequencies_list=[62, 24], use_fast_alg=True).decompose()
        IMFs = PyEMD.EMD(MAX_ITERATION=1000).emd(x)
        C, eps_hat = nonlinear_matching_pursuit(x-IMFs[-1, :], f_s, sparsity=J_true, lambda_init = 0.01, lambda_ = 0.03, K = 20,
                                          show_progress=False, adaptive_rho=True, fixed_V_modes=8,
                                          max_inner_iter=3, auto_init_freq=False,
                                          initial_frequencies_list=[62, 24]).decompose()

        plt.plot(c_true[0, :], label = "First component true")
        plt.plot(C[0, :], label = "First component estimated")
        plt.legend()
        plt.show()

        plt.plot(c_true[1, :], label = "Second component true")
        plt.plot(C[1, :], label = "Second component estimated")
        plt.legend()
        plt.show()

        # plt.plot(c_true[2, :], label = "Third component true")
        # plt.plot(C[2, :], label = "Third component estimated")
        # plt.legend()
        # plt.show()

        plt.plot(eps, label="Noise")
        plt.plot(eps_hat, label="Estimated Noise")
        plt.legend()
        plt.show()

        plt.plot(c_true[2, :], label="Trend")
        plt.plot(IMFs[-1, :], label="Residual")
        plt.legend()
        plt.show()
    if 3 in use:
        IMFs = PyEMD.EMD(MAX_ITERATION=1000).emd(x)
        C, eps_hat = nonlinear_matching_pursuit(x-IMFs[-1, :], f_s, sparsity=J_true, lambda_init = 0.01, lambda_ = 0.3, K = 20,
                                                 show_progress=False, max_inner_iter=1, auto_init_freq=False,
                                                 initial_frequencies_list=[62, 24], use_fast_alg=True).decompose()

        plt.plot(c_true[0, :], label = "First component true")
        plt.plot(C[0, :], label = "First component estimated")
        plt.legend()
        plt.show()

        plt.plot(c_true[1, :], label = "Second component true")
        plt.plot(C[1, :], label = "Second component estimated")
        plt.legend()
        plt.show()

        # plt.plot(c_true[2, :], label = "Third component true")
        # plt.plot(C[2, :], label = "Third component estimated")
        # plt.legend()
        # plt.show()

        plt.plot(eps, label="Noise")
        plt.plot(eps_hat, label="Estimated Noise")
        plt.legend()
        plt.show()

        plt.plot(c_true[2, :], label="Trend")
        plt.plot(IMFs[-1, :], label="Residual")
        plt.legend()
        plt.show()