
'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numpy as np
from ..core import modelcontext, inputvars
from ..model import ObservedRV, TransformedRV
from ..vartypes import discrete_types
from ..blocking import ArrayOrdering, DictToArrayBijection

import theano
from ..theanof import make_shared_replacements, join_nonshared_inputs, CallableTensor, gradient
from theano.tensor import exp, dvector
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict, namedtuple

__all__ = ['advi']

ADVIFit = namedtuple('ADVIFit', 'means, stds, elbo_vals')

def check_discrete_rvs(vars):
    """Check that vars not include discrete variables, excepting ObservedRVs. 
    """
    vars_ = [var for var in vars if not isinstance(var, ObservedRV)]
    if any([var.dtype in discrete_types for var in vars_]):
        raise ValueError('Model should not include discrete RVs for ADVI.')

def advi(vars=None, start=None, model=None, n=5000, accurate_elbo=False, 
    learning_rate=.001, epsilon=.1, seed=None, verbose=1):
    """Run ADVI. 

    Parameters
    ----------
    vars : object
        Random variables. 
    start : Dict or None
        Initial values of parameters (variational means). 
    model : Model
        Probabilistic model. 
    n : int
        Number of interations updating parameters. 
    accurate_elbo : bool
        If true, 100 MC samples are used for accurate calculation of ELBO. 
    learning_rate: float
        Adagrad base learning rate. 
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.  
    seed : int
        Seed to initialize random state. 

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'. 

    'means' and 'stds' include parameters of the variational posterior. 
    """
    seed = seed if type(seed) is int else 12345

    model = modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars
    vars = inputvars(vars)

    check_discrete_rvs(vars)

    n_mcsamples = 100 if accurate_elbo else 1

    # Create variational gradient tensor
    grad, elbo, shared, _ = variational_gradient_estimate(
        vars, model, n_mcsamples=n_mcsamples, seed=seed)

    # Set starting values
    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = ArrayOrdering(vars)
    bij = DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw = np.concatenate([u_start, w_start])

    result, elbos = run_adagrad(uw, grad, elbo, n, learning_rate=learning_rate, epsilon=epsilon, verbose=verbose)

    l = int(result.size / 2)

    u = bij.rmap(result[:l])
    w = bij.rmap(result[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)

def advi_minibatch(vars=None, start=None, model=None, n=5000, n_mcsamples=1, 
    minibatch_RVs=None, minibatch_tensors=None, minibatches=None, total_size=None, 
    learning_rate=.001, epsilon=.1, seed=None, verbose=1):
    """Run mini-batch ADVI. 

    minibatch_RVs, minibatch_tensors and minibatches should be in the 
    same order. 

    Parameters
    ----------
    vars : object
        Random variables. 
    start : Dict or None
        Initial values of parameters (variational means). 
    model : Model
        Probabilistic model. 
    n : int
        Number of interations updating parameters. 
    n_mcsamples : int
        Number of Monte Carlo samples to approximate ELBO. 
    minibatch_RVs : list of ObservedRVs
        Random variables for mini-batch. 
    minibatch_tensors : list of tensors
        Tensors used to create ObservedRVs in minibatch_RVs. 
    minibatches : list of generators
        Generates minibatches when calling next(). 
    totalsize : int
        Total size of training samples. 
    learning_rate: float
        Adagrad base learning rate. 
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.  
    seed : int
        Seed to initialize random state. 

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'. 
    """
    seed = seed if type(seed) is int else 12345

    model = modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars

    vars = set(inputvars(vars)) - set(minibatch_RVs)

    check_discrete_rvs(vars)

    # Create variational gradient tensor
    grad, elbo, shared, uw = variational_gradient_estimate(
        vars, model, minibatch_RVs, minibatch_tensors, total_size, 
        n_mcsamples=n_mcsamples, seed=seed)

    # Set starting values
    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = ArrayOrdering(vars)
    bij = DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw_start = np.concatenate([u_start, w_start])

    shared_inarray = theano.shared(uw_start, 'uw_shared')
    grad = theano.clone(grad, { uw : shared_inarray }, strict=False)
    elbo = theano.clone(elbo, { uw : shared_inarray }, strict=False)
    updates = adagrad(grad, shared_inarray, learning_rate=learning_rate, epsilon=epsilon, n=10)

    # Create in-place update function
    f = theano.function(minibatch_tensors, [shared_inarray, grad, elbo], updates=updates)

    # Run adagrad steps
    elbos = np.empty(n)
    for i in range(n):
        uw_i, g, e = f(*[next(m) for m in minibatches])
        elbos[i] = e
        if verbose and not i % (n//10):
            print('Iteration {0} [{1}%]: ELBO = {2}'.format(i, 100*i//n, e.round(2)))
    
    if verbose:
        print('Finished [100%]: ELBO = {}'.format(elbos[-1].round(2)))

    l = int(uw_i.size / 2)

    u = bij.rmap(uw_i[:l])
    w = bij.rmap(uw_i[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)
    
def run_adagrad(uw, grad, elbo, n, learning_rate=.001, epsilon=.1, verbose=1):
    """Run Adagrad parameter update. 

    This function is only used in batch training. 
    """
    shared_inarray = theano.shared(uw, 'uw_shared')
    grad = CallableTensor(grad)(shared_inarray)
    elbo = CallableTensor(elbo)(shared_inarray)

    updates = adagrad(grad, shared_inarray, learning_rate=learning_rate, epsilon=epsilon, n=10)

    # Create in-place update function
    f = theano.function([], [shared_inarray, grad, elbo], updates=updates)

    # Run adagrad steps
    elbos = np.empty(n)
    for i in range(n):
        uw_i, g, e = f()
        elbos[i] = e
        if verbose and not i % (n//10):
            print('Iteration {0} [{1}%]: ELBO = {2}'.format(i, 100*i//n, e.round(2)))
    
    if verbose:
        print('Finished [100%]: ELBO = {}'.format(elbos[-1].round(2)))
    return uw_i, elbos

def variational_gradient_estimate(
    vars, model, minibatch_RVs=[], minibatch_tensors=[], total_size=None, 
    n_mcsamples=1, seed=None):
    """Calculate approximate ELBO and its (stochastic) gradient. 
    """
    seed = seed if type(seed) is int else 12345

    theano.config.compute_test_value = 'ignore'
    shared = make_shared_replacements(vars, model)

    # Correction sample size 
    r = 1 if total_size is None else \
        float(total_size) / minibatch_tensors[0].shape[0]

    other_RVs = set(model.basic_RVs) - set(minibatch_RVs)
    factors = [r * var.logpt for var in minibatch_RVs] + \
              [var.logpt for var in other_RVs] + model.potentials
    logpt = tt.add(*map(tt.sum, factors))
    
    [logp], inarray = join_nonshared_inputs([logpt], vars, shared)

    uw = dvector('uw')
    uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                        inarray.tag.test_value])

    elbo = elbo_t(logp, uw, inarray, n_mcsamples=n_mcsamples, seed=seed)

    # Gradient
    grad = gradient(elbo, [uw])

    return grad, elbo, shared, uw

def elbo_t(logp, uw, inarray, n_mcsamples, seed):
    """Create Theano tensor of approximate ELBO by Monte Carlo sampling. 
    """
    l = (uw.size/2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    # Callable tensor
    logp_ = lambda input: theano.clone(logp, {inarray: input}, strict=False)

    # Naive Monte-Carlo
    r = MRG_RandomStreams(seed=seed)

    if n_mcsamples == 1:
        n = r.normal(size=inarray.tag.test_value.shape)
        q = n * exp(w) + u
        elbo = logp_(q) + tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))
    else:
        n = r.normal(size=(n_mcsamples, u.tag.test_value.shape[0]))
        qs = n * exp(w) + u
        logps, _ = theano.scan(fn=lambda q: logp_(q),
                               outputs_info=None,
                               sequences=[qs])
        elbo = tt.mean(logps) + tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))

    return elbo

def adagrad(grad, param, learning_rate, epsilon, n, ret_accu=False):
    """Create Theano parameter (tensor) updates by Adagrad. 
    """
    updates = OrderedDict()

    if n == 0:
        updates[param] = param - (-learning_rate * grad)
        accu = None
    else:
        # Compute windowed adagrad using last n gradients
        i = theano.shared(np.array(0), 'i')
        value = param.get_value(borrow=True)
        accu = theano.shared(
            np.zeros(value.shape+(n,), dtype=value.dtype), borrow=True)

        # Append squared gradient vector to accu_new
        accu_new = theano.tensor.set_subtensor(accu[:,i], grad ** 2)
        i_new = theano.tensor.switch((i + 1) < n, i + 1, 0)

        updates[accu] = accu_new
        updates[i] = i_new

        accu_sum = accu_new.sum(axis=1)
        updates[param] = param - (-learning_rate * grad /
                                  theano.tensor.sqrt(accu_sum + epsilon))

    if ret_accu:
        return updates, accu
    else:
        return updates

def logp_t(model, minibatch_scale):
    if minibatch_scale is not None:
        minibatch_RVs = set(minibatch_scale.keys())
        other_RVs = set(model.basic_RVs) - set(minibatch_RVs)
    else:
        minibatch_RVs = []
        other_RVs = model.basic_RVs

    factors = [minibatch_scale[var] * var.logpt for var in minibatch_RVs] + \
              [var.logpt for var in other_RVs] + model.potentials
    logpt = tt.add(*map(tt.sum, factors))

    return logpt

class ADVI(object):
    """Automatic differentiation variational inference step. 

    This class specifies the details of ADVI: the random variables on which the 
    optimization performed, the number of updates for variational parameters, 
    mini-batches fed into the optimization algorithm. 

    Parameters
    ----------
    model : pymc3.Model
        Probabilistic model. 
    vars : list
        List of random variables for which variational posteriors are estimated. 
    vars_update : list
        List of random variables for which variational parameters are updated. 
        All random variables in this should be included in vars. 
    n_iter_grad : int
        Number of parameter updates in each ADVI step. 
    n_mcsamples : int
        Number of Monte Carlo samples to approximate ELBO. 
    minibatch : object
        Mini-batch object. It should have appropriate interface. 
    minibatch_scale : dict \{var : float\}
        Scales of log-probabilities of random variables in the model. 
        That are used to correct the number of samples in each mini-batch. 
        Tentatively, it is set to (# of whole samples) / (mini-batch size).
    learning_rate : float
        Learning rate for Adagrad. 
    epsilon : float
        Adagrad parameter. 
    n_window : int
        Adagrad parameter. 
    seed : int of None
        Seed of random number generator. 
    """
    def __init__(
        self, model=None, start=None, vars=None, vars_update=None, 
        n_iter_advi=10, n_mcsamples=1, minibatch=None, minibatch_scale=None, 
        learning_rate=.001, epsilon=.1, n_window=10, seed=None):
        self.model = model
        self.seed = seed if type(seed) is int else 12345
        self.n_mcsamples = n_mcsamples
        self.minibatch = minibatch
        self.minibatch_scale = minibatch_scale
        self.n_iter_advi = n_iter_advi

        theano.config.compute_test_value = 'ignore'

        model = modelcontext(model)
        if start is None:
            start = model.test_point

        # RVs approximated with variational posteriors
        if vars is None:
            vars = model.vars
        vars = inputvars(vars)

        check_discrete_rvs(vars)

        # RVs updated by this instance
        if vars_update is None:
            vars_update = vars
        vars_update = inputvars(vars_update)

        # RVs not update by this instance
        shared = make_shared_replacements(vars, model)
        for var, share in shared.items():
            share.set_value(start[str(var)])

        # inarray : joined random variables
        logpt = logp_t(model, minibatch_scale)
        [logp], inarray = join_nonshared_inputs([logpt], vars, shared, make_shared=True)

        # Initialize Variational parameters
        ordering = ArrayOrdering(vars)
        bij = DictToArrayBijection(ordering, start)
        u_start = bij.map(start)
        w_start = np.zeros_like(u_start)
        uw_start = np.concatenate([u_start, w_start])
        uw_shared = theano.shared(uw_start, 'uw_shared')

        # Make the mask for gradient vector
        m = np.zeros_like(u_start)
        varnames_update_ = [str(var) for var in vars_update]
        for varname, slc, _, _ in ordering.vmap:
            if varname in varnames_update_:
                m[slc] = 1.
        mask = theano.shared(np.concatenate([m, m]), 'mask')

        # Make tensors of variational parameters
        uw = dvector('uw')
        uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                            inarray.tag.test_value])
        elbo = elbo_t(logp, uw, inarray, n_mcsamples=n_mcsamples, seed=seed)
        grad = gradient(elbo, [uw]) * mask

        # Create in-place update function
        grad = theano.clone(grad, { uw : uw_shared }, strict=False)
        elbo = theano.clone(elbo, { uw : uw_shared }, strict=False)
        updates, accu = adagrad(
            grad, uw_shared, learning_rate=learning_rate, epsilon=epsilon, 
            n=n_window, ret_accu=True)
        f = theano.function([], elbo, updates=updates)

        self.ordering = ordering
        self.shared = shared
        self.uw_shared = uw_shared
        self.inarray = inarray
        self.f = f
        self.accu = accu

    def step(self, point, vparams):
        """Performe ADVI parameter updates. 
        """
        bij = DictToArrayBijection(self.ordering, point)

        # Reset accumulator for Adagrad
        if self.accu is not None:
            self.accu.get_value(borrow=True)[:] = 0.

        # Set random variables not updated by this method
        for var, share in self.shared.items():
            share.container.storage[0] = point[str(var)]

        # Set variational parameters
        l = int(len(self.uw_shared.get_value()) / 2)
        if vparams is not None:
            self.uw_shared.set_value(
                np.hstack((bij.map(vparams['means']), bij.map(vparams['stds'])))
            )

        uw_borrow = self.uw_shared.get_value(borrow=True)

        # Replace observations and variational parameters
        if self.minibatch is not None:
            # Prepare the next mini-batch
            self.minibatch.prepare_next()

            # Set observed values to shared variables
            for t in self.minibatch.observed_tensors:
                t.set_value(self.minibatch.get_observation(t))

            # Set variational parameters of the mini-batch
            for varname, slc, _, _ in self.ordering.vmap:
                if varname in self.minibatch.latent_varnames:
                    u, w = self.minibatch.get_variational_params(varname)
                    uw_borrow[:l][slc] = u
                    uw_borrow[l:][slc] = w

        # Perform ADVI steps
        elbos = []
        for i in range(self.n_iter_advi):
            elbos.append(self.f().ravel())
        elbos = np.array(elbos).ravel()

        # Store variational parameters for latent variables back to minibatch
        if self.minibatch is not None:
            uw = self.uw_shared.get_value()
            u = np.atleast_1d(uw[:l])
            w = np.atleast_1d(uw[l:])
            for varname, slc, shp, dtyp in self.ordering.vmap:
                if varname in self.minibatch.latent_varnames:
                    u_ = u[slc].reshape(shp).astype(dtyp)
                    w_ = w[slc].reshape(shp).astype(dtyp)
                    self.minibatch.set_variational_params(varname, u_, w_)

        uw = self.uw_shared.get_value()
        self.point = point
        vparams = {
            'means': bij.rmap(uw[:l]), 
            'stds': bij.rmap(uw[l:]), 
        }

        return bij.rmap(np.array(self.inarray.get_value())), vparams, elbos

def optimize_vparams(n_iter, steps, start, vparams=None, exp_std=False):
    """Optimize variational parameters. 

    Parameters
    ----------
    n_iter : int
        Number of iterations of ADVI steps. If each ADVI step updates parameters
        n_iter_advi times, the total number of parameter updates is 
        n_iter * n_iter_advi. 
    steps : list
        ADVI steps. 
    start : dict
        Initial values of random variables. These values are ignored for the 
        random variables approximated with variational posteriors. 
    vparams : dict or None (default)
        Variational parameters. 
    exp_std : bool (default to False)
        If true, the returned values of the stds of the variational posteriors 
        are exponentiated. It should be False when using the variational 
        parameters in other calculations (e.g., sample_vp()). 

    Returns
    -------

    """
    elbos = []
    point = start

    for i in range(n_iter):
        for step in steps:
            point, vparams, elbo = step.step(point, vparams)
            elbos.append(elbo)

    u = vparams['means']
    w = vparams['stds']

    if exp_std:
        w = {k: np.exp(v) for k, v in w.items()}

    return ADVIFit(u, w, elbos)

def sample_vpost(n_samples, vars, vparams, seed=1):
    """Draw samples from variational posterior. 

    Parameters
    ----------
    n_samples : int
        Number of random samples. 
    vars : list of random variables
        Random variables for which samples are drawn.
    vparams : dict or ADVIFit
        Variational parameters of the model. It should contain all variational 
        parameters in the model. 
    seed : int
        Seed of random number generator. 

    Returns
    -------
    samples_ : dict
        Random samples. 
    """
    if type(vparams) is ADVIFit:
        vparams = {
            'means': vparams.means, 
            'stds': vparams.stds
        }

    r = MRG_RandomStreams(seed=seed)
    samples = []
    for var in vars:
        var_ = var.transformed if isinstance(var, TransformedRV) else var
        u = theano.shared(vparams['means'][str(var_)]).ravel()
        w = theano.shared(vparams['stds'][str(var_)]).ravel()
        n = r.normal(size=u.tag.test_value.shape)
        var = theano.clone(var, {var_: (n * tt.exp(w) + u).reshape(var_.tag.test_value.shape)})
        samples.append(var)
    f = theano.function([], samples)
    
    samples = []
    for i in range(n_samples):
        samples.append(f())
        
    samples_ = {}
    for i, var in enumerate(vars):
        samples_.update({str(var): np.stack([sample[i] for sample in samples], axis=0)})
        
    return samples_
