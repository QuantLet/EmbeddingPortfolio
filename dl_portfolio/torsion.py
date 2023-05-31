"""
Cf: https://github.com/HarperGuo/Risk-Parity-and-Beyond
Created on Sat May 13 01:30:50 2017
@author: Hao Guo
"""

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import sqrtm

from dl_portfolio.weights import equal_class_weights


def EffectiveBets(w, Sigma, t):
    Sigma = np.asmatrix(Sigma)
    w = np.asmatrix(w)
    p = np.asmatrix(np.asarray(np.linalg.inv(t.T) * w.T) * np.asarray(
        t * Sigma * w.T)) / (w * Sigma * w.T)
    enb = np.exp(- p.T * np.log(p + 1e-6))
    return p, enb


def torsion(Sigma, model, method='exact', max_niter=10000):
    Sigma = np.asmatrix(Sigma)
    n = Sigma.shape[0]
    if model == 'pca':
        eigval, eigvec = np.linalg.eig(Sigma)
        idx = np.argsort(-eigval)
        t = eigvec[:,idx]
    elif model == 'minimum-torsion':
        # C: correlation matrix
        sigma = np.sqrt(np.diag(Sigma))
        C = np.asmatrix(np.diag(1.0/sigma)) * np.asmatrix(Sigma) * \
            np.asmatrix(np.diag(1.0/sigma))
        # Riccati root of correlation matrix
        c = sqrtm(C)
        if method == 'approximate':
            t = (np.asmatrix(sigma) / np.asmatrix(c)) * np.asmatrix(
                np.diag(1.0/sigma))
        elif method == 'exact':
            # initialize
            d = np.ones((n))
            f = np.zeros((max_niter))
            # iterating
            for i in range(max_niter):
                U = np.asmatrix(np.diag(d)) * c * c * np.asmatrix(np.diag(d))
                u = sqrtm(U)
                q = np.linalg.inv(u) * np.asmatrix(np.diag(d)) * c
                d = np.diag(q * c)
                pi = np.asmatrix(np.diag(d)) * q
                f[i] = np.linalg.norm(c - pi, 'fro')
                # if converge
                if i > 0 and abs(f[i]-f[i-1])/f[i] <= 1e-6:
                    f = f[0:i]
                    break
                elif i == max_niter and abs(
                        f[max_niter]-f[max_niter-1])/f[max_niter] >= 1e-6:
                    print(f'number of max iterations reached: n_iter = {str(max_niter)}')
            x = pi * np.linalg.inv(np.asmatrix(c))
            t = np.asmatrix(np.diag(sigma)) * x * np.asmatrix(
                np.diag(1.0/sigma))
    return t


def get_min_torsion_bets(cv_results, cv_port_weights,
                         market_budget: pd.DataFrame, level: str):
    assets = market_budget.index.tolist()
    d = len(assets)
    portfolios = list(cv_port_weights[0].keys()) + ["equal", "equal_class"]
    n_bets = {p: pd.DataFrame() for p in portfolios}
    rcs = {p: [] for p in portfolios}

    dates = [cv_results[0][cv]["returns"].index[0] for cv in
             cv_results[0].keys()]

    for i in cv_results.keys():
        if i > 0:
            portfolios = ["aerp", "rb_factor"]

        i_n_bets = {p: [] for p in portfolios}
        i_rcs = {p: [] for p in portfolios}
        for cv in cv_results[i].keys():
            scale_ = cv_results[i][cv]["scaler"]["attributes"]["scale_"]
            if level == "asset":
                ret = pd.concat(
                    [
                        cv_results[i][cv]["train_returns"],
                        cv_results[i][cv]["returns"]
                    ]
                )
                Sigma = np.cov(ret.T)
            elif level == "factor":
                ret = pd.concat(
                    [
                        cv_results[i][cv]["train_returns"],
                        cv_results[i][cv]["returns"]
                    ]
                ).values
                input_ = ((ret -
                           cv_results[i][cv]["scaler"]["attributes"]["mean_"]
                           ) / scale_)
                factors = np.dot(input_,
                                 cv_results[i][cv]["embedding"])
                W_tilde = np.dot(
                    np.diag(scale_),
                    cv_results[i][cv]["loading"]
                )
                V_tilde = scipy.linalg.null_space(W_tilde.T).T
                res_factors = np.dot(np.linalg.pinv(V_tilde).T, ret.T).T
                all_factors = np.concatenate([factors, res_factors],
                                             axis=1)
                Sigma = np.cov(all_factors.T)
            t_mt = torsion(Sigma, 'minimum-torsion', method='exact')
            for p in portfolios:
                if p == "equal":
                    a = np.ones(d) / d
                elif p == "equal_class":
                    assert market_budget is not None
                    a = equal_class_weights(market_budget).loc[
                        assets].values
                else:
                    a = cv_port_weights[cv][p].values

                if level == "factor":
                    # Get factor weights
                    a = np.concatenate(
                        [np.dot(W_tilde.T, a), np.dot(V_tilde, a)])

                rc_i, nb = EffectiveBets(a, Sigma, t_mt)
                nb = np.real(np.array(nb)[0, 0])
                rc_i = np.asarray(rc_i)[:, 0]
                i_n_bets[p].append(nb)
                i_rcs[p].append(rc_i)

        i_n_bets = {p: pd.Series(i_n_bets[p]) for p in portfolios}
        i_rcs = {p: np.array(i_rcs[p]) for p in portfolios}
        for p in portfolios:
            n_bets[p][i] = i_n_bets[p]
            rcs[p].append(pd.DataFrame(i_rcs[p], index=dates))

    for p in n_bets:
        n_bets[p] = n_bets[p].mean(1)

    n_bets = pd.DataFrame(n_bets)
    n_bets.index = dates

    return rcs, n_bets
