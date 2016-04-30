import numpy as np
from pymc3.model import TransformedRV
from ..core import inputvars

class ADVIMinibatch(object):
    """Mini-batch for ADVI. 

    This class prepares a buffer for variational parameters of the given 
    latent variables. 

    Parameters
    ----------
    observed_data : dict
        Pairs of tensors and ndarrays. All ndarrays should have the 
        same size (total size of data). 
    minibatch_size : int
        Size of mini-batches. It should be a divisor of the total data size. 
    seed : int or None
        Seed of random number generator.
    """
    def __init__(
        self, observed_data, latent_vars, minibatch_size, transpose_vars=[], 
        seed=None):
        self.observed_tensors = observed_data.keys()
        self.latent_vars = [] if latent_vars is None else inputvars(latent_vars)
        self.latent_varnames = [str(v) for v in self.latent_vars]
        self.data = observed_data
        self.total_size = self.data[self.data.keys()[0]].shape[0]
        self.minibatch_size = minibatch_size
        self.transpose_varnames = [str(v) for v in inputvars(transpose_vars)]
        self.rng = np.random.RandomState(seed)
        self.ixs = None

        # Check total size of data
        assert(all([v.shape[0] == self.total_size for v in self.data.values()]))

        # Initialize variational parameters
        self.vparams = {}
        for var in self.latent_vars:
            dsize = var.transformed.dsize if type(var) is TransformedRV \
                    else var.dsize
            sample_size = dsize / self.minibatch_size
            u = np.zeros((self.total_size, sample_size))
            w = np.zeros((self.total_size, sample_size))
            self.vparams.update({str(var): (u, w)})

    def prepare_next(self):
        """Prepare next mini-batch. 
        """
        pass

    def get_observation(self, tensor):
        return self.data[tensor][self.ixs]

    def get_variational_params(self, varname):
        u, w = self.vparams[varname]

        transpose = varname in self.transpose_varnames
        u_ = u[self.ixs].T if transpose else u[self.ixs]
        w_ = w[self.ixs].T if transpose else w[self.ixs]

        return u_.ravel(), w_.ravel()

    def set_variational_params(self, varname, u, w):
        u_, w_ = self.vparams[varname]

        transpose = varname in self.transpose_varnames

        u_[self.ixs] = u.T if transpose else u
        w_[self.ixs] = w.T if transpose else w

class RandomizedMinibatch(ADVIMinibatch):
    def prepare_next(self):
        """Prepare next mini-batch. 
        """
        self.ixs = self.rng.permutation(self.total_size)[:self.minibatch_size]

class SequentialMinibatch(ADVIMinibatch):
    def prepare_next(self):
        """Prepare next mini-batch. 
        """
        if self.ixs is None:
            self.ixs = np.arange(self.minibatch_size).astype(int)
        else:
            start = self.ixs[-1] + 1
            start = 0 if self.total_size == start else start

            if (start + self.minibatch_size) <= self.total_size:
                self.ixs = np.arange(
                    start, start + self.minibatch_size, dtype=int)
            else:
                ixs1 = np.arange(start, self.total_size, dtype=int)
                ixs2 = np.arange(
                    0, self.minibatch_size - (self.total_size - start), 
                    dtype=int)
                self.ixs = np.hstack((ixs1, ixs2))
