import numpy as np
import random
from pandas import DataFrame
try:
    from tqdm import tqdm
except:
    tqdm = None

class walker:
    def __init__(self, scale=1, tune_interval=100):
        self.scale = scale
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0
    def step(self, q0):
        if self.steps_until_tune == 0:
            acc_rate = self.accepted / self.tune_interval
            if acc_rate < 0.001:
                self.scale *= 0.1
            elif acc_rate < 0.05:
                self.scale *= 0.5
            elif acc_rate < 0.2:
                self.scale *= 0.9
            elif acc_rate > 0.95:
                self.scale *= 10.0
            elif acc_rate > 0.75:
                self.scale *= 2.0
            elif acc_rate > 0.5:
                self.scale *= 1.1
            self.steps_until_tune = self.tune_interval
            self.accepted = 0
        self.steps_until_tune -= 1
        q = q0 + np.random.normal() * self.scale
        return q
    def accept(self, lnp0, lnp):
        if lnp == -np.inf:
            return False
        if -np.inf < np.log(np.random.uniform()) < lnp - lnp0:
            self.accepted += 1
            return True
        return False

class Sampler:
    def __init__(self, q0, lnprob, args=(), scale=None, tune_interval=None, progress_bar=True):
        self.progress_bar = progress_bar
        self.lnprob = lnprob
        self.args = args
        self.walkers = []
        self.q = np.array(q0, dtype=np.float64)
        self.has_blobs = False
        self.lnp0 = self.lnprob(self.q, *self.args)
        try:
            self.lnp0, blob = self.lnp0
            self.has_blobs = True
        except TypeError:
            pass
        if scale is None:
            scale = [1 for i in self.q]
        if tune_interval is None:
            tune_interval = [100 for i in self.q]
        for i, _ in enumerate(self.q):
            self.walkers.append(walker(scale[i], tune_interval[i]))
    def run(self, nsamples, burnin=0):
        samples = np.empty((nsamples, self.q.size), dtype=np.float64)
        blobs = []
        if tqdm is None or not self.progress_bar:
            iter_samples = range(nsamples + burnin)
        else:
            iter_samples = tqdm(range(nsamples + burnin))
        for i in iter_samples:
            if i < burnin:
                self.sample()
            else:
                if self.has_blobs:
                    samples[i-burnin, :], blob = self.sample()
                    blobs.append(blob)
                else:
                    samples[i-burnin, :] = self.sample()
        if self.has_blobs:
            return samples, blobs
        else:
            return samples
    def sample(self):
        for i, walker in enumerate(self.walkers):
            q0 = self.q[i]
            self.q[i] = walker.step(q0)
            lnp = self.lnprob(self.q, *self.args)
            if self.has_blobs:
                lnp, blob = lnp
            if walker.accept(self.lnp0, lnp):
                self.lnp0 = lnp
            else:
                self.q[i] = q0
        if self.has_blobs:
            return self.q, blob
        else:
            return self.q

#def get_logp(out_label, gr4, warmup, peq, lnprob_prior, area_up, area_down, q_kde=None):
#    def logp(xs):
#        '''
#        Parameters:
#            - xs: dict of the form label:parameter_list
#        '''
#        lnp = 0
#        for x in xs.values():
#            for i, v in enumerate(x):
#                lnp += lnprob_prior[i](v)
#            if not np.isfinite(lnp):
#                return -np.inf, np.full_like(peq.p.values, np.inf)
#
#        q_sim = 0
#        for label, x in {k:v for k, v in xs.items() if k != out_label}.items():
#            # upstream basins
#            # x includes the delay in the GR model
#            q_sim += gr4(x).run([peq[f'p{label}'].values, peq[f'e{label}'].values]) * area_up[label]
#        # downstream basin
#        q_sim += gr4(xs[out_label]).run([peq[f'p{out_label}'].values, peq[f'e{out_label}'].values]) * area_down
#        q_sim /= sum(area_up.values()) + area_down
#        if q_kde is None:
#            # observation is measured water level
#            h_obs = peq.h_obs.values
#            h_err = peq.h_err.values
#            h_sim = np.hstack((np.full(warmup, np.nan), dist_map(q_sim[warmup:], h_obs[warmup:])))
#            df = DataFrame({'h_sim': h_sim, 'h_obs': h_obs, 'h_err': h_err})[warmup:].dropna()
#            std2 = df.h_err * df.h_err
#            # must not have zero error on observation
#            min_std2 = np.max(std2) / 100
#            std2 = np.clip(std2, min_std2, None)
#            return lnp + np.sum(-np.square(df.h_sim.values - df.h_obs.values) / (2 * std2) - np.log(np.sqrt(2 * np.pi * std2))), q_sim
#        else:
#            # observation is simulated streamflow
#            lnp_q = 0
#            for i in range(warmup, q_sim.size, sim_step):
#                lnp_q += lnprob_from_density(q_kde[:, :, i])(q_sim[i])
#            return lnp + lnp_q, q_sim
#    return lnprob

def uniform_density(a, b, nb=100):
    xy = np.empty((2, nb))
    xy[0] = np.linspace(a, b, nb)
    xy[1, :] = 1
    xy[1] /= np.trapz(xy[1], x=xy[0])
    return xy

def lnprob_from_density(xy, vmin=None, vmax=None):
    def lnprob(value):
        # the simple way, everything outside the possible values has 0 probability
        # return np.log(np.interp(value, xy[0], xy[1], left=0, right=0))
        if ((vmin is not None) and (value < vmin)) or ((vmax is not None) and (value > vmax)):
            return -np.inf
        if xy[0][0] <= value <= xy[0][-1]:
            v = np.interp(value, xy[0], xy[1])
        else:
            # tail distribution, probability must not be 0 and must decrease
            if value < xy[0][0]:
                vtail = xy[1][0]
                e = value - xy[0][0] # negative
            else:
                vtail = xy[1][-1]
                e = xy[0][-1] - value # negative
            e /= xy[0][-1] - xy[0][0] # normalization
            v = vtail * np.exp(e)
        return np.log(v)
    return lnprob

def dist_map(x, y):
    df = DataFrame({'x': x, 'y': y}).dropna()
    x_sorted = np.sort(df.x.values)
    y_sorted = np.sort(df.y.values)
    return np.interp(x, x_sorted, y_sorted)

def get_kde(samples, a=np.inf, b=-np.inf, nb=100):
    '''Computes the Kernel Densit Estimate.

    Parameters:
        - samples: array of samples

    Returns:
        - xyz: a list consisting of sample values, probability density of the sample values, and samples.
    '''
    samples = samples[np.isfinite(samples)]
    smin, smax = np.min(samples), np.max(samples)
    if smin == smax:
        smin *= 0.99
        smax *= 1.01
        samples[:2] = [smin, smax]
    if a < smin:
        smin = a
    if b > smax:
        smax = b
    xyz = []
    xyz.append(np.linspace(smin, smax, nb))
    xyz.append(stats.gaussian_kde(samples)(xyz[0]))
    xyz[1] /= np.trapz(xyz[1], x=xyz[0])
    xyz.append(samples)
    return xyz

def sample(population, n=1):
    '''Samples from a population.

    Parameters:
        - population: the population from which to sample from.

    Returns:
        - samples: sample(s) from the population.
    '''
    samples = [population[random.randint(0, len(population)-1)] for _ in range(n)]
    if n == 1:
        return samples[0]
    else:
        return samples
