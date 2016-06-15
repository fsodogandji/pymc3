
'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numpy as np
from ..core import modelcontext, inputvars, ArrayOrdering, DictToArrayBijection
# from ..model import ObservedRV
# from ..vartypes import discrete_types

import theano
from ..theanof import reshape_t
import theano.tensor as tt
from collections import OrderedDict
from .advi import check_discrete_rvs, elbo_t, adagrad, ADVIFit

__all__ = ['advi_minibatch']

# Flatten list
from itertools import chain
flt = lambda l: list(chain.from_iterable(l))

class Encoder(object):
    """Encode vector into latent representation.
    """
    def encode(self):
        """Returns variational mean and std vectors. 
        """
        pass
        
    def get_params(self):
        """Returns list of parameters (shared variables) of the encoder. 
        """
        pass

def _value_error(cond, str):
    if not cond:
        raise ValueError(str)

def _replace_shared_minibatch_tensors(minibatch_tensors):
    """Replace shared variables in minibatch tensors with normal tensors. 
    """
    givens = dict()
    tensors = list()

    for t in minibatch_tensors:
        if isinstance(t, theano.compile.sharedvalue.SharedVariable):
            t_ = t.type()
            tensors.append(t_)
            givens.update({t: t_})
        else:
            tensors.append(t)

    return tensors, givens

def _join_RVs(global_RVs, global_order, local_RVs, local_order):
    joined_global = tt.concatenate([v.ravel() for v in global_RVs])
    uw_global = tt.dvector('uw_global')
    uw_global.tag.test_value = np.concatenate([joined_global.tag.test_value, 
                                               joined_global.tag.test_value])

    if local_RVs is not None:
        joined_local = tt.concatenate([v.ravel() for v in local_RVs])
        uw_local = tt.dvector('uw_local')
        uw_local.tag.test_value = np.concatenate([joined_local.tag.test_value, 
                                                  joined_local.tag.test_value])
        joined = tt.concatenate([joined_global, joined_local])
        rvs = [v for v in global_RVs] + [v for v in local_RVs]
    else:
        uw_local = None
        joined = joined_global
        rvs = [v for v in global_RVs]

    tensor_type = joined.type
    inarray = tensor_type('concat_1d_rvs')
    inarray.tag.test_value = joined.tag.test_value

    get_var = {var.name : var for var in rvs}

    replace = {
        get_var[var] : reshape_t(inarray[slc], shp).astype(dtyp)
        for var, slc, shp, dtyp in global_order.vmap 
    }

    if local_RVs is not None:
        assert(joined_global.ndim == 1)
        l = joined_global.tag.test_value.shape[0]
        inarray_local = inarray[l:]
        replace.update({
            get_var[var] : reshape_t(inarray_local[slc], shp).astype(dtyp)
            for var, slc, shp, dtyp in local_order.vmap 
        })

    return replace, inarray, uw_global, uw_local

def _make_elbo_t(
    global_RVs, local_RVs, observed_RVs, global_order, local_order, model, 
    minibatch_tensors=[], n_mcsamples=1, random_seed=20090425):
    """Calculate approximate ELBO and its (stochastic) gradient. 
    """
    # Scale log probability for mini-batches 
    factors = [s * v.logpt for v, s in observed_RVs.items()] + \
              [v.logpt for v in global_RVs] + model.potentials
    if local_RVs is not None:
        factors += [s * v.logpt for v, (_ , s) in local_RVs.items()]
    logpt = tt.add(*map(tt.sum, factors))

    # Replace RVs with 1d vector
    replace, concat_1d_rvs, uw_global, uw_local = _join_RVs(
        global_RVs, global_order, local_RVs, local_order
    )
    logp = theano.clone(logpt, replace, strict=False)

    # ELBO
    if uw_local is None:
        uw = uw_global
    else:
        l_uw_global = (uw_global.size / 2).astype('int64')
        l_uw_local = (uw_local.size / 2).astype('int64')
        uw = tt.concatenate([uw_global[:l_uw_global], # variational mean, global
                             uw_local[:l_uw_local],   # variational mean, local
                             uw_global[l_uw_global:], # variational std, global
                             uw_local[l_uw_local:]])  # variational std, local

    elbo = elbo_t(logp, uw, concat_1d_rvs, n_mcsamples, random_seed)

    return elbo, uw_global, uw_local

def _check_minibatches(minibatch_tensors, minibatches):
    _value_error(isinstance(minibatch_tensors, list), 
                 'minibatch_tensors should be a list.')

    _value_error(isinstance(minibatches, list), 
                 'minibatches should be a list.')

    _value_error(len(minibatch_tensors) == len(minibatches), 
                 'len(minibatch_tensors) should be equal to ' +
                 'len(minibatches')

def _get_rvss(
    minibatch_RVs, local_RVs, observed_RVs, minibatch_tensors, total_size):
    """Returns local_RVs and observed_RVs. 

    This function is used for backward compatibility of how input arguments are 
    given. 
    """
    if minibatch_RVs is not None:
        _value_error(isinstance(minibatch_RVs, list), 
                     'minibatch_RVs should be a list.')

        _value_error((local_RVs is None) and (observed_RVs is None), 
                     'When minibatch_RVs is given, local_RVs and ' +
                     'observed_RVs should be None.')

        s = np.float32(total_size) / minibatch_tensors[0].shape[0]
        local_RVs = OrderedDict()
        observed_RVs = OrderedDict([(v, s) for v in minibatch_RVs])

    else:
        _value_error((isinstance(local_RVs, OrderedDict) and 
                      isinstance(observed_RVs, OrderedDict)), 
                     'local_RVs and observed_RVs should be OrderedDict.')

    return local_RVs, observed_RVs

def advi_minibatch(vars=None, start=None, model=None, n=5000, n_mcsamples=1, 
    minibatch_RVs=None, minibatch_tensors=None, minibatches=None, 
    local_RVs=None, observed_RVs=None, encoder_params=[], 
    total_size=None, scales=None, learning_rate=.001, epsilon=.1, 
    random_seed=20090425, verbose=1):
    """Run mini-batch ADVI. 

    minibatch_tensors and minibatches should be in the same order. 

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
    total_size : int
        Total size of training samples. 
    learning_rate: float
        Adagrad base learning rate. 
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.  
    random_seed : int
        Seed to initialize random state. 

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'. 
    """
    theano.config.compute_test_value = 'ignore'

    model = modelcontext(model)
    vars = inputvars(vars if vars is not None else model.vars)
    start = start if start is not None else model.test_point
    check_discrete_rvs(vars)
    _check_minibatches(minibatch_tensors, minibatches)

    # For backward compatibility in how input arguments are given
    local_RVs, observed_RVs = _get_rvss(minibatch_RVs, local_RVs, observed_RVs, 
                                        minibatch_tensors, total_size)

    # Replace local_RVs with transformed variables
    ds = model.deterministics
    get_transformed = lambda v: v if v not in ds else v.transformed
    local_RVs = OrderedDict(
        [(get_transformed(v), (uw, s)) for v, (uw, s) in local_RVs.items()]
    )

    # Get global variables
    rvs = lambda x: [rv for rv in x]
    global_RVs = list(set(vars) - set(rvs(local_RVs) + rvs(observed_RVs)))

    # Ordering for concatenation of random variables
    global_order = ArrayOrdering([v for v in global_RVs])
    local_order = ArrayOrdering([v for v in local_RVs])

    # ELBO wrt variational parameters
    elbo, uw_global, uw_local = _make_elbo_t(
        global_RVs, local_RVs, observed_RVs, global_order, local_order, 
        model, minibatch_tensors, n_mcsamples, random_seed
    )

    # Variational parameters for global RVs
    start = {v.name: start[v.name] for v in global_RVs}
    bij = DictToArrayBijection(global_order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw_start = np.concatenate([u_start, w_start])
    uw_global_shared = theano.shared(uw_start, 'uw_global_shared')

    # Variational parameters for local RVs, encoded from samples in mini-batches
    uws = [uw for _, (uw, _) in local_RVs.items()]
    uw_local_encoded = tt.concatenate([uw[0].ravel() for uw in uws] + 
                                      [uw[1].ravel() for uw in uws])

    # Replace tensors in ELBO
    updates = {uw_global: uw_global_shared, 
               uw_local: uw_local_encoded}
    elbo = theano.clone(elbo, updates, strict=False)
    # from theano.gof.graph import inputs
    # print([v for v in inputs([elbo]) if not (isinstance(v, tt.TensorConstant) or isinstance(v, tt.Constant))])
    # import pdb; pdb.set_trace() # debug

    # Replace input shared variables with tensors
    isshared = lambda t: isinstance(t, theano.compile.sharedvalue.SharedVariable)
    tensors = [(t.type() if isshared(t) else t) for t in minibatch_tensors]
    updates = {t: t_ for t, t_ in zip(minibatch_tensors, tensors)
               if isshared(t)}
    elbo = theano.clone(elbo, updates, strict=False)

    # Parameter updates
    params = [uw_global_shared] + encoder_params
    updates = {}
    for param in params:
        g = tt.grad(elbo, wrt=param)
        updates.update(adagrad(g, param, learning_rate, epsilon, n=10))

    # Create in-place update function
    f = theano.function(tensors, elbo, updates=updates)

    # Run adagrad steps
    elbos = np.empty(n)
    for i in range(n):
        e = f(*[next(m) for m in minibatches])
        elbos[i] = e
        if verbose and not i % (n//10):
            if not i:
                print('Iteration {0} [{1}%]: ELBO = {2}'.format(i, 100*i//n, e.round(2)))
            else:
                avg_elbo = elbos[i-n//10:i].mean()
                print('Iteration {0} [{1}%]: Average ELBO = {2}'.format(i, 100*i//n, avg_elbo.round(2)))
    
    if verbose:
        print('Finished [100%]: ELBO = {}'.format(elbos[-1].round(2)))

    l = int(uw_global_shared.get_value(borrow=True).size / 2)

    u = bij.rmap(uw_global_shared.get_value(borrow=True)[:l])
    w = bij.rmap(uw_global_shared.get_value(borrow=True)[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)
