# From lopez de prado

# On 20151231 by MLdP <lopezdeprado@lbl.gov>
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import prado.cla.CLA as CLA
from prado.hrp import correlDist, getIVP, getQuasiDiag, getRecBipart


# ------------------------------------------------------------------------------
def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data: each row is a variable
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))

    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F,
                                      size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])

    # 4) add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])

    return x, cols


def get_cluster_mapper(size0, nvars, cols):
    cluster_labels = np.unique(cols).tolist()
    col_cluster_mapper = {i: i for i in range(nvars)}
    for i, c in enumerate(range(size0, nvars)):
        col_cluster_mapper[c] = cols[i]
    col_cluster_mapper = pd.Series(col_cluster_mapper)
    cluster_mapper = {
        c: col_cluster_mapper.index[col_cluster_mapper == c].tolist() for c
        in cluster_labels
    }
    return cluster_mapper, col_cluster_mapper


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1, 1)  # Not used by C portf
    lB = np.zeros(mean.shape)
    uB = np.ones(mean.shape)
    cla = CLA.CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


def worker(steps, methods_mapper, nObs=520, size0=5, size1=5, mu0=0,
           sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22):

    if steps % 10 == 0:
        print(f"Steps to go: {steps}")

    stats = {i: pd.Series() for i in methods_mapper}
    # 1) Prepare data for one experiment
    x, _ = generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
    r = {i: pd.Series() for i in methods_mapper}
    pointers = range(sLength, nObs, rebal)

    # 2) Compute portfolios in-sample
    for pointer in pointers:
        x_ = x[pointer - sLength:pointer]
        cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)

        # 3) Compute performance out-of-sample
        x_ = x[pointer:pointer + rebal]
        for func_name in methods_mapper:
            w_ = methods_mapper[func_name](cov=cov_, corr=corr_)  # callback
            r_ = pd.Series(np.dot(x_, w_))
            r[func_name] = r[func_name].append(r_)

    # 4) Evaluate and store results
    for func_name in methods_mapper:
        r_ = r[func_name].reset_index(drop=True)
        p_ = (1 + r_).cumprod()
        stats[func_name] = p_.iloc[-1] - 1  # terminal return

    return stats


def hrpMC(n_jobs=1, numIters=1e4, nObs=520, size0=5, size1=5, mu0=0,
          sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22):
    # Monte Carlo experiment on HRP
    methods_mapper = {
        "getIVP": getIVP,
        "getHRP": getHRP
    }
    stats = {i: pd.Series() for i in methods_mapper}

    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            stats_iters = _parallel_pool(
                delayed(worker)(
                    numIters - numIter, methods_mapper, nObs=nObs,
                    size0=size0, size1=size1, mu0=mu0, sigma0=sigma0,
                    sigma1F=sigma1F, sLength=sLength, rebal=rebal
                )
                for numIter in range(numIters)
            )
        for numIter in range(numIters):
            for func_name in stats:
                stats[func_name].loc[numIter] = stats_iters[numIter][func_name]
    else:
        for numIter in range(numIters):
            stats_iter = worker(numIters - numIter, methods_mapper, nObs=nObs,
                        size0=size0, size1=size1,  mu0=mu0, sigma0=sigma0,
                        sigma1F=sigma1F, sLength=sLength, rebal=rebal)
            for func_name in stats:
                stats[func_name].loc[numIter] = stats_iter[func_name]

    # 5) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    stats.to_csv('stats.csv')
    df0, df1 = stats.std(), stats.var()
    print(stats)
    print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1))
    return


if __name__=='__main__':
    hrpMC(n_jobs=8, numIters=10)
