import numpy as np
from pymc3.model import TransformedRV 

class RandomizedMinibatch(object):
    """Mini-batch with random sampling. 

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
        self.latent_vars = [] if latent_vars is None else latent_vars
        self.data = observed_data
        self.total_size = self.data[self.data.keys()[0]].shape[0]
        self.minibatch_size = minibatch_size
        self.transpose_vars = transpose_vars
        self.rng = np.random.RandomState(seed)

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
            self.vparams.update({var: (u, w)})

    def prepare_next(self):
        """Prepare next mini-batch. 
        """
        self.ixs = self.rng.permutation(self.total_size)[:self.minibatch_size]

    def get_observation(self, tensor):
        return self.data[tensor][self.ixs]

    def get_variational_params(self, var):
        u, w = self.vparams[var]

        transpose = var in self.transpose_vars
        u_ = u[self.ixs].T if transpose else u[self.ixs]
        w_ = w[self.ixs].T if transpose else w[self.ixs]

        return u_.ravel(), w_.ravel(-1)

    def set_variational_params(self, var, u, w):
        u_, w_ = self.vparams[var]

        transpose = var in self.transpose_vars

        u_[self.ixs] = u.T if transpose else u
        w_[self.ixs] = w.T if transpose else w
