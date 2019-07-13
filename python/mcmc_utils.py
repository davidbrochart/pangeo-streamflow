import numpy as np
import random
from pandas import DataFrame
from mcmc.dist import logp_from_pdf

def get_likelihood_logp(gr4, warmup, pe, area, he=None, q_pdf=None, sim_step=1):
    '''
    Parameters
    ----------
    pe : list
    area : list
    '''
    def likelihood_logp(x_flat):
        '''
        Parameters
        ----------
        x_flat : list
            A list of parameters, where the last ones are downstream's.

        Returns
        -------
        lp : float
            The log probability.
        '''
        # upstream basins
        x_up = [x_flat[i:i+5] for i in range(0, len(x_flat)-4, 5)]
        q_sim = sum([gr4(x).run(pe[i]) * area[i] for i, x in enumerate(x_up)])
        # downstream basin
        q_sim += gr4(x_flat[-4:]).run(pe[-1]) * area[-1]
        q_sim /= sum(area)
        if q_pdf is None:
            # observation is measured water level
            h_obs = he.h.values
            h_err = he.e.values
            h_sim = np.hstack((np.full(warmup, np.nan), dist_map(q_sim.values[warmup:], h_obs[warmup:])))
            df = DataFrame({'h_sim': h_sim, 'h_obs': h_obs, 'h_err': h_err}).dropna()
            std2 = (df.h_err.values * df.h_err.values) * 10 # this factore to take into account model/data uncertainty
            # must not have zero error on observation
            min_std2 = np.max(std2) / 100
            std2 = np.clip(std2, min_std2, None)
            lp = np.sum(-np.square(df.h_sim.values - df.h_obs.values) / (2 * std2) - np.log(np.sqrt(2 * np.pi * std2)))
        else:
            # observation is simulated streamflow
            lp = sum([logp_from_pdf(q_pdf[:, :, i], q_sim.values[i]) for i in range(warmup, len(q_sim), sim_step)])
        return lp, q_sim
    return likelihood_logp

def get_prior_logp(x_pdf):
    '''
    This assumes the parameters are independant.
    '''
    def prior_logp(values):
        return sum([logp_from_pdf(pdf, v) for pdf, v in zip(x_pdf, values)])
    return prior_logp

def dist_map(x, y):
    df = DataFrame({'x': x, 'y': y}).dropna()
    x_sorted = np.sort(df.x.values)
    y_sorted = np.sort(df.y.values)
    return np.interp(x, x_sorted, y_sorted)
