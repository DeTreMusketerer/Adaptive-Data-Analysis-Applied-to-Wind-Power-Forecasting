# -*- coding: utf-8 -*-
"""

Created on Fri Nov 19 11:23:03 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this module the main functionality supporting the use of seasonal
autoregressive integrated moving average models as described in the report
        Forecasting Wind Power Production
            - Chapter 2: Time Series Analysis

This model class can be used to estimate parameters of s-ARIMAX and s-VARIMAX
models using the scripts
    - sARIMAX_validation
    - sARIMAX_test
    - sVARIMAX_validation
    - sVARIMAX_test

The module has been developed using Python 3.9 with the
libraries numpy and scipy.

"""

import numpy as np
from time import time


class sVARMAX(object):
    """

    Main class for estimation and forecasting of s-VARIMAX(p, d, q) X (p_s, d_s, q_s)_s
    models using OLS for quick and dirty fit of parameters.

    Details regarding the models can be found in the report
        Forecasting Wind Power Production
            - Chapter 2: Time Series Analysis
            - Chapter 6: Experimental Setup
                - Section 6.2.2: s-ARIMAX
                - Section 6.2.4: s-VARIMAX

    """
    def __init__(self, y, z_reg, z_NWP, missing_t, p=1, d=0, q=0, p_s=0, q_s=0, s=288,
                 m=0, m_s=0, l="all", use_NWP=True, use_reg=True):
        """

        Parameters
        ----------
        y : ndarray, size=(n, k)
            Power data.
        z_reg : ndarray, size=(n, 2)
            Regulation data.
        z_NWP : ndarray, size=(55, n_nwp, 11*k)
            Numerical weather prediction data.
        missing_t : ndarray
            Array of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the array is
            zero and the last entry in the list is n.
        p : int, optional
            Autoregressive order. The default is 1.
        d : int
            Order of differencing. Options are 0 and 1. The default is 0.
        q : int
            Moving average order. The default is 0.
        p_s : int, optional
            Seasonal autoregressive order. The default is 0.
        q_s : int, optional
            Seasonal moving average order. The default is 0.
        s : int, optional
            Seasonal delay. The default is 288.
        m : int, optional
            Order of autoregressive model used for the initial parameter
            estimate. The default is 0.
        m_s : int, optional
            Seasonal order for the autoregressive model used for the initial
            parameter estimate. The default is 0.
        l : int, optional
            Sub-grid. Options are l = 0, \dots, 20. This input should be given
            in the case of a univariate time series model. The default is "all".
        use_NWP : bool, optional
            Boolean variable to decide if numerical weather predictions (NWPs)
            should be used as exogenous variables. The default is True.
        use_reg : bool, optional
            Boolean variable to decide if down-regulation
            should be used as an exogenous variable. The default is True.

        """
        # Initialize variables
        self.use_reg = use_reg
        self.use_NWP = use_NWP

        self.p = p
        self.d = d
        self.q = q
        self.p_s = p_s
        self.q_s = q_s
        self.s = s
        self.l = l
        self.m = m
        self.m_s = m_s
        if p_s == 0: assert m_s == 0
        assert d == 0 or d == 1

        # Store data
        self.y, self.missing_t = self.do_differencing(y.astype(dtype=np.float32), missing_t)
        self.nr_missing_t = len(self.missing_t)-1
        if self.use_reg:
            self.z_reg = z_reg.astype(dtype=np.float32)
        else:
            self.z_reg = None
        if self.use_NWP:
            self.z_NWP = z_NWP.astype(dtype=np.float32)
        else:
            self.z_NWP = None

        # Initialize more variables
        self.n, self.k = np.shape(self.y)
        if self.use_NWP is True and self.use_reg is True:
            self.r_part = 13
        elif self.use_NWP is True and self.use_reg is False:
            self.r_part = 12
        elif self.use_NWP is False and self.use_reg is True:
            self.r_part = 2
        elif self.use_NWP is False and self.use_reg is False:
            self.r_part = 1
        else:
            raise AssertionError("Invalid input(s) supplied to use_NWP and/or use_reg")
        self.r = self.k*self.r_part

        self.max_delay_AR = p_s*s + p # The maximum delay in the autoregressive part
        self.max_delay_MA = q_s*s + q # The maximum delay in the moving average part
        self.max_delay = max(self.max_delay_AR, self.max_delay_MA) # The maximum delay
        self.p_tot = p+(p+1)*p_s # Total number of autoregressive delays
        self.q_tot = q+(q+1)*q_s # Total number of moving average delays

        # Choose exogenous variable method
        if self.use_NWP is False:
            self.make_z = self.make_EMD_z
        elif self.use_NWP is True and self.use_reg is True:
            self.make_z = self.make_NWP_z
        else:
            raise AssertionError("This case is not supported.")

    def do_differencing(self, y, missing_t):
        """
        Differencing for the power data. Note that differencing for the
        exogenous variables are done in make_NWP_z().

        Parameters
        ----------
        y : ndarray, size=(n, 21)
            Wind power data.
        missing_t : array
            Array of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the array is
            zero and the last entry is n.

        Returns
        -------
        y : ndarray, size(n, k)
            Differenced wind power data.
        missing_t : array
            Array of time indices where a discontinuity in time is present due
            to missing power history data for the differenced data.
            The first entry in the array is zero and the last entry is n.

        """
        if self.d == 0:
            return y, missing_t
            # if self.l != "all":
            #     power = np.expand_dims(y[:, self.l], -1)
            #     y = power
        elif self.d == 1:
            # if self.l != "all":
            #     power = np.expand_dims(y[:, self.l], -1)
            # elif self.l == "all":
            #     power = y
            y = y[1:, :] - y[:-1, :]
            missing_t[1:] = missing_t[1:]-1
            return y, missing_t

    def update_parameters(self, Theta):
        """

        Updates the parameters using the dictionary Theta.

        Save to self
        -------
        Psi, Psi, Xi, Sigma_u

        """
        self.Phi = Theta["Phi"]
        if self.q != 0 or self.q_s != 0:
            self.Psi = Theta["Psi"]
        else:
            self.Psi = []
        self.Xi = Theta["Xi"]
        self.Sigma_u = Theta["Sigma_u"]

    def return_parameters(self):
        return self.Phi, self.Psi, self.Xi, self.Sigma_u

    def fit(self):
        """

        Conduct parameter estimation using OLS (quick and dirty).

        """
        self.z = np.zeros((self.n, self.r))
        for t in range(self.n):
            self.z[t, :] = self.make_z(0, t, self.z_reg, self.z_NWP).astype(dtype=np.float32)

        if self.q == 0:
            Xi, Phi, Sigma_u, _ = self.sVARX_fit(self.p, self.p_s, self.s)
            Theta = {"Phi": Phi, "Psi": None, "Xi": Xi, "Sigma_u": Sigma_u}
        else:
            # Do initial parameter estimation
            Xi, Phi, Psi, Sigma_u = self.sVARMAX_fit(self.m, self.m_s)
            Theta = {"Phi": Phi, "Psi": Psi, "Xi": Xi, "Sigma_u": Sigma_u}
        self.update_parameters(Theta)

    def sVARX_fit(self, p, p_s, s):
        """

        Fit a s-VARX(p) x (p_s)_s using OLS.

        Parameters
        ----------
        p : int
            Autoregressive order.
        p_s : int
            Seasonal autoregressive order.
        s : int
            Seasonal delay.

        Returns
        -------
        Xi : ndarray, size=(k, r)
            Exogenous variable parameter matrix.
        Phi : list, len=(p+(p+1)*p_s)
            List of autoregressive parameter matrices given as
            ndarrays of size=(k, k).
        Sigma_u : ndarray, size=(k, k)
            Covariance of white noise process.
        u_hat : ndarray, size=(n, k)
            Estimate of white noise process.

        """

        if self.p_s == 0 and self.q_s == 0:
            if self.d == 0 and self.l == "all":
                print(f"Fit a VARX({p}) model.")
            elif self.d == 1 and self.l == "all":
                print(f"Fit a VARIMAX({p}, {self.d}, {0}) model.")
            elif self.d == 0 and self.l != "all":
                print(f"Fit a ARX({p}) model.")
            elif self.d == 1 and self.l != "all":
                print(f"Fit a ARIMAX({p}, {self.d}, {0}) model.")
            else:
                raise AssertionError("Invalid differencing input. Options are d={0, 1}.")
        else:
            if self.d == 0 and self.l == "all":
                print(f"Fit a s-VARX({p}) x ({p_s})_{s} model.")
            elif self.d == 1 and self.l == "all":
                print(f"Fit a s-VARIMAX({p}, {self.d}, {0}) x ({p_s}, {0}, {0})_{s} model.")
            elif self.d == 0 and self.l != "all":
                print(f"Fit a s-ARX({p}) x ({p_s})_{s} model.")
            elif self.d == 1 and self.l != "all":
                print(f"Fit a s-ARIMAX({p}, {self.d}, {0}) x ({p_s}, {0}, {0})_{s} model.")
            else:
                raise AssertionError("Invalid differencing input. Options are d={0, 1}.")

        delay_list_AR = [j_s*s+j for j_s in range(p_s+1) for j in range(p+1)][1:]
        max_delay_AR = p_s*s + p
        p_tot = p+(p+1)*p_s

        u_hat_temp = np.zeros((self.n-2*288-(max_delay_AR*self.nr_missing_t), self.k), dtype=np.float32)
        pars = np.zeros((p_tot*self.k+self.r_part, self.k), dtype=np.float32)

        idx_list = []

        if self.l == "all":
            iter_l = 0
        else:
            iter_l = self.l

        for l in range(iter_l, iter_l+self.k):
            idx = 0
            Y = np.zeros(self.n-2*288-(max_delay_AR*self.nr_missing_t), dtype=np.float32)
            X = np.zeros((self.n-2*288-(max_delay_AR*self.nr_missing_t), p_tot*self.k+self.r_part), dtype=np.float32)
            for missing_t_idx in range(self.nr_missing_t):
                idx_list.append(idx)
                a = self.missing_t[missing_t_idx]+max_delay_AR
                if missing_t_idx < self.nr_missing_t-1:
                    b = self.missing_t[missing_t_idx+1]-288
                else:
                    b = self.missing_t[missing_t_idx+1]
                for t in range(a, b):
                    X_t = np.zeros((p_tot, self.k))
                    for counter, delay in enumerate(delay_list_AR):
                        X_t[counter, :] = self.y[t-delay, :]
                    X[idx, :p_tot*self.k] = X_t.flatten()
                    if self.k == 1:
                        X[idx, p_tot*self.k:] = self.z[t, :]
                        Y[idx] = self.y[t, 0]
                    elif self.k == 21:
                        X[idx, p_tot*self.k:] = self.z[t, l*self.r_part:(l+1)*self.r_part]
                        Y[idx] = self.y[t, l]
                    idx += 1
            idx_list.append(idx)
            if self.k == 1:
                pars[:, 0], u_hat_temp[:, 0] = self.multivariate_OLS(Y, X)
            elif self.k == 21:
                pars[:, l], u_hat_temp[:, l] = self.multivariate_OLS(Y, X)
        zeros = np.zeros((max_delay_AR+288, self.k), dtype=np.float32)
        u_hat = np.concatenate((np.zeros((max_delay_AR, self.k)), u_hat_temp[idx_list[0]:idx_list[1], :]), axis=0)
        u_hat = np.concatenate((u_hat, zeros, u_hat_temp[idx_list[1]:idx_list[2], :]), axis=0)
        u_hat = np.concatenate((u_hat, zeros, u_hat_temp[idx_list[2]:idx_list[3], :]), axis=0)

        Phi = [pars[j*self.k:(j+1)*self.k, :] for j in range(p_tot)]
        Xi = np.zeros((self.k, self.r), dtype=np.float32)
        if self.k == 1:
            Xi[0, :] = pars[p_tot*self.k:, 0]
        elif self.k == 21:
            for l in range(self.k):
                Xi[l, l*self.r_part:(l+1)*self.r_part] = pars[p_tot*self.k:, l]
        Sigma_u = np.sum(np.array([np.outer(u_hat[t, :], u_hat[t, :]) for t in range(self.n-2*288-(max_delay_AR*self.nr_missing_t))]), axis=0)/(self.n-2*288-(max_delay_AR*self.nr_missing_t)-1)
        return Xi, Phi, Sigma_u, u_hat

    def sVARMAX_fit(self, m, m_s):
        """

        Fit s-VARMAX using OLS.

        1) Fit a s-VARX(m) x (m_s)_s for m >> p and m_s >> p_s model to y
           using OLS. Compute the residuals u_hat for the resulting model.
        2) Using u_hat do OLS to estimate the s-VARMAX(p, q) x (p_s, q_s)_s
           parameters.

        Parameters
        ----------
        m : int
            Autoregressive order for the s-VARX(m) x (m_s)_s in step 1).
        m_s : int
            Seasonal autoregressive order for the s-VARX(m) x (m_s)_s in step 1).

        Returns
        -------
        Xi : ndarray, size=(k, r)
            Exogenous variable parameter matrix.
        Phi : list, len=(p_tot)
            List of autoregressive parameter matrices given as
            ndarrays of size=(k, k).
        Psi : list, len=(q_tot)
            List of moving average parameter matrices given as
            ndarrays of size=(k, k).
        Sigma_u : ndarray, size=(k, k)
            Covariance of white noise process.

        """
        if self.p_s != 0: assert self.s > m

        # Step 1)
        _, _, _, u_hat = self.sVARX_fit(m, m_s, self.s)

        if self.p_s == 0 and self.q_s == 0:
            if self.l == "all":
                print(f"Fit a VARIMAX({self.p}, {self.d}, {self.q}) model.")
            elif self.l != "all":
                print(f"Fit a ARIMAX({self.p}, {self.d}, {self.q}) model.")
        else:
            if self.l == "all":
                print(f"Fit a s-VARIMAX({self.p}, {self.d}, {self.q}) x ({self.p_s}, {0}, {self.q_s})_{self.s} model.")
            elif self.l != "all":
                print(f"Fit a s-ARIMAX({self.p}, {self.d}, {self.q}) x ({self.p_s}, {0}, {self.q_s})_{self.s} model.")

        # Step 2)
        delay_list_AR = [j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)][1:]
        delay_list_MA = [i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)][1:]

        pars = np.zeros(((self.p_tot+self.q_tot)*self.k + self.r_part, self.k), dtype=np.float32)
        u_hat_new = np.zeros((self.n-2*288-(self.max_delay_AR*self.nr_missing_t), self.k), dtype=np.float32)

        if self.l == "all":
            iter_l = 0
        else:
            iter_l = self.l

        for l in range(iter_l, iter_l+self.k):
            idx = 0
            Y = np.zeros(self.n-2*288-(self.max_delay_AR*self.nr_missing_t), dtype=np.float32)
            X = np.zeros((self.n-2*288-(self.max_delay_AR*self.nr_missing_t), (self.p_tot+self.q_tot)*self.k + self.r_part), dtype=np.float32)
            for missing_t_idx in range(self.nr_missing_t):
                a = self.missing_t[missing_t_idx]+self.max_delay_AR
                if missing_t_idx < self.nr_missing_t-1:
                    b = self.missing_t[missing_t_idx+1]-288
                else:
                    b = self.missing_t[missing_t_idx+1]
                for t in range(a, b):
                    X_t_AR = np.zeros((self.p_tot, self.k), dtype=np.float32)
                    X_t_MA = np.zeros((self.q_tot, self.k), dtype=np.float32)
                    for counter, delay_AR in enumerate(delay_list_AR):
                        X_t_AR[counter, :] = self.y[t-delay_AR, :]
                    for counter, delay_MA in enumerate(delay_list_MA):
                        X_t_MA[counter, :] = -u_hat[t-delay_MA, :]
                    X[idx, :(self.p_tot+self.q_tot)*self.k] = np.vstack((X_t_AR, X_t_MA)).flatten()
                    if self.k == 1:
                        X[idx, (self.p_tot+self.q_tot)*self.k:] = self.z[t, :]
                        Y[idx] = self.y[t, 0]
                    elif self.k == 21:
                        X[idx, (self.p_tot+self.q_tot)*self.k:] = self.z[t, l*self.r_part:(l+1)*self.r_part]
                        Y[idx] = self.y[t, l]
                    idx += 1
            if self.k == 1:
                pars[:, 0], u_hat_new[:, 0] = self.multivariate_OLS(Y, X)
            elif self.k == 21:
                pars[:, l], u_hat_new[:, l] = self.multivariate_OLS(Y, X)
        Phi = [pars[j*self.k:(j+1)*self.k, :] for j in range(self.p_tot)]
        Psi = [pars[self.p_tot*self.k+i*self.k:self.p_tot*self.k+(i+1)*self.k, :] for i in range(self.q_tot)]
        Xi = np.zeros((self.k, self.r), dtype=np.float32)
        if self.k == 1:
            Xi[0, :] = pars[(self.p_tot+self.q_tot)*self.k:, 0]
        elif self.k == 21:
            for l in range(self.k):
                Xi[l, l*self.r_part:(l+1)*self.r_part] = pars[(self.p_tot+self.q_tot)*self.k:, l]
        Sigma_u = np.sum(np.array([np.outer(u_hat_new[t, :], u_hat_new[t, :]) for t in range(self.n-2*288-(self.max_delay_AR*self.nr_missing_t))]), axis=0)/(self.n-2*288-(self.max_delay_AR*self.nr_missing_t)-1)
        return Xi, Phi, Psi, Sigma_u

    def multivariate_OLS(self, Y, X):
        """

        Compute OLS for a multivariate regression problem.

        Parameters
        ----------
        Y : ndarray, size=(n, k)
            Target.
        X : ndarray, size=(n, r)
            Design matrix.

        Returns
        -------
        B : ndarray, size=(r, k)
            Parameter matrix.
        eps : ndarray, size=(n, k)
            Residuals.

        """
        t1 = time()
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        t2 = time()
        print("Parameter fit time: {}".format(t2-t1))
        eps = Y - X @ B
        return B, eps

    def estimate_noise(self, tau_ahead, y_test, z_NWP_test, z_reg_test, test_missing_t):
        """
        Parameters
        ----------
        tau_ahead : int
            Compute and test on up to tau-ahead forecasts.
        y_test : ndarray, size=(n_test, k)
            Endogenous variable.
        z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
            Numerical weather predictions with the given transformations.
            The first axis is the tau_ahead axis while the second axis gives
            a new set of NWP. The last axis is as follows:
                (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                 WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
        z_reg_test : ndarray, size=(n_test, 2)
            Regulation data for DK1 in the first column and DK2 in the second
            column.
        test_missing_t : list
            List of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the list is
            zero and the last entry in the list is n.

        Returns
        -------
        u_hat : ndarray, size=(n_test, k)
            Noise process.
        """
        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        u_hat = np.zeros((n_test, self.k))
        for missing_t_idx in range(test_nr_missing_t):
            a = test_missing_t[missing_t_idx]+self.max_delay
            if missing_t_idx < test_nr_missing_t-1:
                b = test_missing_t[missing_t_idx+1]-tau_ahead-288
            else:
                b = test_missing_t[missing_t_idx+1]-tau_ahead
            for t in range(a, b):
                u_hat[t, :] += y_test[t, :]
                for j, idx in enumerate(array_AR):
                    u_hat[t, :] -= np.dot(self.Phi[j], y_test[t-idx, :])
                for i, idx in enumerate(array_MA):
                    u_hat[t, :] += np.dot(self.Psi[i], u_hat[t-idx, :])
                z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
                u_hat[t, :] -= np.dot(self.Xi, z_data)
        return u_hat

    def forecast(self, tau_ahead, y_test, z_reg_test, z_NWP_test,
                 test_missing_t, P_test):
        """

        Compute the tau-ahead forecast using the truncated forecasting as
        defined in property 3.7 of Shumway2017.

        Parameters
        ----------
        tau_ahead : int
            Compute tau-ahead forecast t+tau given time t-1.
        y_test : ndarray, size=(n_test, k)
            Endogenous variable.
        z_reg_test : ndarray, size=(n_test, 2)
            Regulation data for DK1 in the first column and DK2 in the second
            column.
        z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
            Numerical weather predictions with the given transformations.
            The first axis is the tau_ahead axis while the second axis gives
            a new set of NWP. The last axis is as follows:
                (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                 WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
        test_missing_t : list
            List of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the list is
            zero and the last entry in the list is n.
        P_test : ndarray, size=(n_test+1, k), optional
            Wind power at time t-2. Used when first order differencing is used.

        Returns
        -------
        P_bar : ndarray, size=(t_end-t_start, k)
            Wind power forecast.
        idx_list : list
            List containing the indices for which forecasts are made. This
            is needed due to the missing data.

        """

        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        y_bar = np.zeros((tau_ahead, n_test, self.k))
        if self.d == 1:
            P_bar = np.zeros((tau_ahead, n_test, self.k))

        phi_mat = np.hstack(self.Phi)
        if self.q != 0 or self.q_s != 0:
            psi_mat = np.hstack(self.Psi)
            beta = np.concatenate((phi_mat, -psi_mat, self.Xi), axis=1)
        else:
            beta = np.concatenate((phi_mat, self.Xi), axis=1)

        u_hat = self.estimate_noise(tau_ahead, y_test, z_NWP_test,
                                    z_reg_test, test_missing_t)

        idx_list = []

        for tau_i in range(tau_ahead):
            if tau_i % 20 == 0:
                print("Tau ahead: {}".format(tau_i))
            for missing_t_idx in range(test_nr_missing_t):
                a = test_missing_t[missing_t_idx]+self.max_delay
                if missing_t_idx < test_nr_missing_t-1:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead-288
                else:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead
                for t in range(a, b):
                    if tau_i == 0:
                        idx_list.append(t)
                        z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
                        y_vec = y_test[t-array_AR, :].flatten()
                        u_vec = u_hat[t-array_MA, :].flatten()
                        data_vec = np.hstack((y_vec, u_vec, z_data))
                        y_bar[0, t, :] = np.dot(beta, data_vec)
                    else:
                        bar_AR = array_AR[tau_i-array_AR >= 0]
                        test_AR = array_AR[tau_i-array_AR < 0]
                        hat_MA = array_MA[tau_i-array_MA < 0]
                        if len(bar_AR) != 0:
                            y_vec_bar = y_bar[tau_i-bar_AR, t, :].flatten()
                        else:
                            y_vec_bar = np.array([])
                        if len(test_AR) != 0:
                            y_vec_test = y_test[t+tau_i-test_AR, :].flatten()
                        else:
                            y_vec_test = np.array([])
                        if len(hat_MA) != 0:
                            u_vec = u_hat[t+tau_i-hat_MA, :].flatten()
                        else:
                            u_vec = np.array([])
                        y_bar[tau_i, t, :] += np.dot(phi_mat, np.hstack((y_vec_bar, y_vec_test)))
                        if self.q != 0 or self.q_s != 0:
                            y_bar[tau_i, t, :] -= np.dot(psi_mat[:, (len(array_MA)-len(hat_MA))*self.k:], u_vec)
                        z_data = self.make_z(tau_i, t, z_reg_test, z_NWP_test)
                        y_bar[tau_i, t, :] += np.dot(self.Xi, z_data)

                    if self.d == 1:
                        if tau_i == 0:
                            P_bar[0, t, :] = y_bar[0, t, :] + P_test[t, :]
                        else:
                            P_bar[tau_i, t, :] =  y_bar[tau_i, t, :] + P_bar[tau_i-1, t, :]
                    else:
                        P_bar = y_bar
        return P_bar, idx_list

    def test(self, tau_ahead, y_test, z_reg_test, z_NWP_test, test_missing_t, P_max, P_cap):
        """

        Test function. This function assumes the parameters have been fitted
        using self.fit(). The functionality mainly relies on self.forecast().

        Parameters
        ----------
        tau_ahead : int
            Compute tau-ahead forecast t+tau given time t-1.
        y_test : ndarray, size=(n_test, k)
            Endogenous variable.
        z_reg_test : ndarray, size=(n_test, 2)
            Regulation data for DK1 in the first column and DK2 in the second
            column.
        z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
            Numerical weather predictions with the given transformations.
            The first axis is the tau_ahead axis while the second axis gives
            a new set of NWP. The last axis is as follows:
                (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                 WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
        test_missing_t : list
            List of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the list is
            zero and the last entry in the list is n.
        P_max : ndarray, size=(k,)
            Maximum wind power measured in the training data for each wind
            area.
        P_cap : ndarray, size(k,)
            Installed capacity in each wind area.

        Returns
        -------
        MSE : ndarray, size=(tau_ahead, k)
            Mean squared error of wind power forecast.
        NMAE : ndarray, size=(tau_ahead, k)
            Normalised mean absolute error of wind power forecast.
        eps : ndarray, size=(tau_ahead, n_test, k)
            Residuals of wind power forecast.

        """
        print("Commence testing...")
        assert tau_ahead >= 1 and tau_ahead < 55*12

        # Store data
        P_test = np.copy(y_test)
        y_test, test_missing_t = self.do_differencing(y_test.astype(dtype=np.float32), test_missing_t)
        n_test, _ = y_test.shape
        if self.use_reg:
            z_reg_test = z_reg_test.astype(dtype=np.float32)
        else:
            z_reg_test = None
        if self.use_NWP:
            z_NWP_test = z_NWP_test.astype(dtype=np.float32)
        else:
            z_NWP_test = None

        P_bar, idx_list = self.forecast(tau_ahead, y_test, z_reg_test, z_NWP_test,
                                        test_missing_t, P_test)

        idx_array = np.array(idx_list)
        #eps = np.zeros((tau_ahead, len(idx_list), self.k))
        eps = np.zeros((tau_ahead, n_test, self.k))
        for tau_i in range(tau_ahead):
            if self.d == 0:
                eps[tau_i, idx_array, :] = P_bar[tau_i, idx_array, :] - P_test[idx_array+tau_i, :]
            elif self.d == 1:
                eps[tau_i, idx_array, :] = P_bar[tau_i, idx_array, :] - P_test[idx_array+tau_i+1, :]

        print(f"Residual shape: {eps.shape}")
        MSE = np.mean(eps[:, idx_array, :]**2, axis=1)
        NRMSE = np.sqrt(np.mean(eps[:, idx_array, :]**2, axis=1)/(P_cap**2))
        NB = np.mean(eps[:, idx_array, :], axis=1)/(P_cap)
        NMAE = np.mean(np.abs(eps[:, idx_array, :]), axis=1)/P_cap
        return MSE, NRMSE, NB, NMAE, eps

    def make_NWP_z(self, tau_i, t, z_reg, z_NWP):
        """
        Function to make a z_data vector in a numpy array given a time
        (tau_i, t) and the exogenous variables z_reg and z_NWP as well as 
        the order of differencing d.

        Parameters
        ----------
        tau_i : int
            tau_i-ahead prediction index.
        t : int
            Time index.
        z_reg : ndarray, size=(n, 2)
            Regulations data ordered as (dk1, dk2).
        z_NWP : ndarray, size=(n, 11*k)
            Numerical weather prediction variables post transformations.

        Returns
        -------
        z_data : ndarray, size=(13*k,)
            Exogenous variable vector.
        """
        if self.d == 1:
            t += 1
        nwp_t, remainder1 = divmod(t, 36)
        nwp_tau_i, remainder2 = divmod(remainder1+tau_i, 12)
        z_data = np.zeros(self.r)
        if self.d == 0:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1]
                z_data[2:] = z_NWP[nwp_tau_i, nwp_t, self.l*11:(self.l+1)*11]
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1], 6)
                nwp_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                z_data[nwp_list] = z_NWP[nwp_tau_i, nwp_t, :]
        elif self.d == 1:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1]
                if remainder2 == 0 and nwp_tau_i != 0:
                    z_data[2:] = z_NWP[nwp_tau_i, nwp_t, self.l*11:(self.l+1)*11] - z_NWP[nwp_tau_i-1, nwp_t, self.l*11:(self.l+1)*11]
                elif remainder2 == 0 and nwp_tau_i == 0 and remainder1 == 0:
                    z_data[2:] = z_NWP[0, nwp_t, self.l*11:(self.l+1)*11] - z_NWP[2, nwp_t-1, self.l*11:(self.l+1)*11]
                else:
                    z_data[2:] = np.zeros(11)
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1], 6)
                nwp_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                if remainder2 == 0 and nwp_tau_i != 0:
                    z_data[nwp_list] = z_NWP[nwp_tau_i, nwp_t, :] - z_NWP[nwp_tau_i-1, nwp_t, :]
                elif remainder2 == 0 and nwp_tau_i == 0 and remainder1 == 0:
                    z_data[nwp_list] = z_NWP[0, nwp_t, :] - z_NWP[2, nwp_t-1, :]
                else:
                    z_data[nwp_list] = np.zeros(11*21)
        return z_data

    def make_EMD_z(self, tau_i, t, z_reg, z_NWP=None):
        """
        Function to make a z_data vector in a numpy array given a time
        (tau_i, t) and the exogenous variables z_reg. The exogenous variables
        consists of a 1 giving a bias term, and the down-regulation.

        Parameters
        ----------
        tau_i : int
            tau_i-ahead prediction index.
        t : int
            Time index.
        z_reg : ndarray, size=(n, 2)
            Regulations data ordered as (dk1, dk2).
        z_NWP :
            Dummy variable to make make_EMD_a and make_NWP_z compatible.

        Returns
        -------
        z_data : ndarray, size=(2,)
            Exogenous variable vector.
        """
        if self.d != 0: raise AssertionError("Differencing is implemented for this case.")
        z_data = np.zeros(self.r)
        z_data[0] = 1
        if self.use_reg is True:
            if self.l < 15:
                z_data[1] = z_reg[t+tau_i, 0]
            else:
                z_data[1] = z_reg[t+tau_i, 1]
        return z_data

