
import numpy as np
import pandas as pd

import scipy
from scipy.stats.mstats import mquantiles
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib import rcParams

from classy import Class

import emcee

from abc import ABC, abstractmethod

# initialize class
cosmo = Class()

# Planck 2018 parameters from https://arxiv.org/pdf/1807.06209.pdf
parameters = {'omega_b': 0.02233,
              'gauge' : "newtonian",
              'omega_cdm': 0.1198,
              'h': 0.6737,
              'A_s':2.097E-9,
              'n_s':0.9652,
              'tau_reio':0.0540}

# set the class parameters
cosmo.set({'output':'mPk',
           "P_k_max_1/Mpc":10.0,
           **parameters})

# run class
cosmo.compute()
h=parameters["h"]



class Data:

    def __init__(self,
                 k,
                 P_k,
                 P_k_error,
                 cosmology):

        self.k = k
        self.P_k = P_k
        self.P_k_error = P_k_error
        self.cosmo = cosmology

        h = cosmo.h()
        self.P_k_lcdm = [self.cosmo.pk(_k ,0 ) * h**3 for _k in self.k]

        self.T_k = np.sqrt(self.P_k /self.P_k_lcdm)
        self.T_k_error = 0.5 *self.P_k_error /np.sqrt(self.P_k_lcdm * self.P_k)

    def plot_Pk(self, logscale=True, color="k"):
        plt.errorbar(self.k, self.P_k, self.P_k_error, color=color ,marker="o", ms=5, ls="")

        plt.xlabel("k [h/Mpc]")
        plt.ylabel("$P_k [Mpc/h^3]$")

        if logscale:
            plt.xscale("log")
            plt.yscale("log")

    def plot_Tk(self, logscale=True, color="k"):
        plt.errorbar(self.k, self.T_k, self.T_k_error, color=color, marker="o", ms=5, ls="")

        plt.xlabel("k [h/Mpc]")
        plt.ylabel("$T_k $")

        if logscale:
            plt.xscale("log")


class Sampler:
    """sampler for Tk posterior"""

    def __init__(self,
                 data,
                 model,
                 param_names,
                 param_limits,
                 guess=[0, 0, 0],
                 n_walkers=30,
                 n_steps=10000):

        self.model = model
        self.data = data
        self.param_names = param_names
        self.param_limits = param_limits
        self.data_params = data.k, data.T_k, data.T_k_error

        self.guess = guess

        self.n_walkers = n_walkers
        self.n_dim = len(model.model_params)
        self.n_steps = n_steps

    def log_likelihood(self, model_params, data_params, ):
        """log of gaussian likelihood """
        # alpha, beta, gamma = model_params
        k, T_k, T_k_error = data_params

        model_params_dict = {key: value for (key, value) in zip(self.param_names, model_params)}
        T_k_model = self.model.get_T_k(model_params_dict)

        sigma2_model = T_k_error ** 2  # + y_model ** 2 * np.exp(2 * log_f)
        return -0.5 * np.sum((T_k - T_k_model) ** 2 / sigma2_model)

    def neg_log_likelihood(self, model_params, data_params):
        return -self.log_likelihood(model_params, data_params)

    def log_prior(self, model_params):
        """log of flat prior"""

        if all([limit[0] < model_param < limit[1] for limit, model_param in
                zip(self.param_limits, model_params)]):
            return 0.0

        return -np.inf

    def log_probability(self, model_params, data_params):

        lp = self.log_prior(model_params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(model_params, data_params)

    def sample(self, find_lkl_min=True):
        """sample the posterior"""
        if find_lkl_min:
            # find the maximum likelihood (min of -log_lkl)
            print("finding log_lik_min...")
            log_lik_min = minimize(self.neg_log_likelihood, self.guess, args=(self.data_params,))

            # distribute the walkers around the maximum likelihood
            # taken from the emcee tutorial
            pos_0 = log_lik_min.x + 1e-3 * np.random.randn(self.n_walkers, self.n_dim)

        else:
            pos_0 = self.guess + 1e-3 * np.random.randn(self.n_walkers, self.n_dim)

        print("initializing the sampler...")
        # start sampling the posterior
        # with Pool() as pool:
        sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.log_probability,
                                        args=(self.data_params,),
                                        # pool = pool,
                                        )
        sampler.run_mcmc(pos_0, self.n_steps, progress=True)

        self.sampler = sampler
        self.samples = sampler.get_chain()

    def plot(self):

        labels = self.param_names
        n_params = len(self.param_names)
        fig, axes = plt.subplots(n_params, figsize=(3 * n_params, 7), sharex=True)

        labels = self.param_names
        for i in range(self.n_dim):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");




# %%

class AbstractModel(ABC):

    def __init__(self,
                 k,
                 model_params,
                 cosmology=None,
                 ):

        assert type(model_params) is dict

        # set the k at which the model power spectrum should be calculated
        self.k = k
        self.model_params = model_params

        # calculate the transfer function according to the model
        self.T_k = self.get_T_k(model_params)

        # calculate the base (lamdaCMB power spectrum)
        # since this is saved as a list because it needs to be calcuated once
        if cosmology is not None:
            self.cosmology = cosmo
            h = cosmo.h()
            self.P_k_lcdm = [cosmo.pk(_k, 0) * h ** 3 for _k in self.k]
            self.P_k = self.get_P_k()

    @abstractmethod
    def get_T_k(self, model_params):
        """place holder """

        pass

    def get_P_k(self):
        """# Eq. 1 in Murgia+2018"""
        assert hasattr(self.k, "__len__"), "k has to be a list or array"
        assert self.cosmology is not None

        return [self.T_k[i] ** 2 * self.P_k_lcdm[i] for i, _ in enumerate(self.k)]

    def plot_Pk(self, T_k=False, logscale=True, *args, **kwargs):

        plt.plot(self.k, self.P_k, "tab:red", lw=1, *args, **kwargs)

        plt.xlabel("k [h/Mpc]")
        plt.ylabel("$P_k$")

        if logscale:
            plt.xscale("log")
            plt.yscale("log")

    def plot_Tk(self, T_k=False, logscale=True, *args, **kwargs):

        plt.plot(self.k, self.T_k, "tab:red", lw=1, *args, **kwargs)

        plt.xlabel("k [h/Mpc]")
        plt.ylabel("$T_k$")

        if logscale:
            plt.xscale("log")

