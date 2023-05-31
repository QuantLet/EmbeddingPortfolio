"""
Overwrite HCPortfolio from riskfolio: Rewrite class to accept optimal number of
cluster

Copyright (c) 2020-2022, Dany Cajas
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Riskfolio-Lib nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pandas as pd
import numpy as np

import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform

import riskfolio.src.AuxFunctions as af
import riskfolio.src.DBHT as db
import riskfolio.src.GerberStatistic as gs
from riskfolio.src.HCPortfolio import HCPortfolio as RFHCPortfolio
import riskfolio.src.ParamsEstimation as pe


class HCPortfolio(RFHCPortfolio):
    def __init__(
            self,
            returns=None,
            alpha=0.05,
            a_sim=100,
            beta=None,
            b_sim=None,
            kappa=0.30,
            solver_rl=None,
            solvers=None,
            w_max=None,
            w_min=None,
            alpha_tail=0.05,
            gs_threshold=0.5,
            bins_info="KN",
    ):
        super().__init__(returns=returns, alpha=alpha, a_sim=a_sim,
                         beta=beta, b_sim=b_sim, kappa=kappa,
                         solver_rl=solver_rl, solvers=solvers, w_max=w_max,
                         w_min=w_min, alpha_tail=alpha_tail,
                         gs_threshold=gs_threshold, bins_info=bins_info)

    # Overwirte hierarchical clustering with optimal number of cluster
    def _hierarchical_clustering(
            self,
            model="HRP",
            linkage="ward",
            codependence="pearson",
            max_k=10,
            optimal_n_cluster=None,
            leaf_order=True,
    ):
        if optimal_n_cluster is not None:
            assert max_k is None, (
                "You cannot pass both max_k and optimal_n_cluster"
            )
        # Calculating distance
        if codependence in {
            "pearson",
            "spearman",
            "kendall",
            "gerber1",
            "gerber2",
            "custom_cov",
        }:
            dist = np.sqrt(np.clip((1 - self.codep) / 2, a_min=0.0, a_max=1.0))
        elif codependence in {"abs_pearson", "abs_spearman", "abs_kendall",
                              "distance"}:
            dist = np.sqrt(np.clip((1 - self.codep), a_min=0.0, a_max=1.0))
        elif codependence in {"mutual_info"}:
            dist = af.var_info_matrix(self.returns, self.bins_info).astype(
                float)
        elif codependence in {"tail"}:
            dist = -np.log(self.codep).astype(float)

        # Hierarchical clustering
        dist = dist.to_numpy()
        dist = pd.DataFrame(dist, columns=self.codep.columns,
                            index=self.codep.index)
        if linkage == "DBHT":
            # different choices for D, S give different outputs!
            D = dist.to_numpy()  # dissimilarity matrix
            if codependence in {
                "pearson",
                "spearman",
                "kendall",
                "gerber1",
                "gerber2",
                "custom_cov",
            }:
                codep = 1 - dist ** 2
                S = codep.to_numpy()  # similarity matrix
            else:
                S = self.codep.to_numpy()  # similarity matrix
            (_, _, _, _, _, clustering) = db.DBHTs(
                D, S, leaf_order=leaf_order
            )  # DBHT clustering
        else:
            p_dist = squareform(dist, checks=False)
            clustering = hr.linkage(p_dist, method=linkage,
                                    optimal_ordering=leaf_order)

        if optimal_n_cluster:
            k = optimal_n_cluster
        else:
            if model in {"HERC", "HERC2", "NCO"}:
                # optimal number of clusters
                k = af.two_diff_gap_stat(self.codep, dist, clustering, max_k)
            else:
                k = None

        return clustering, k

    def optimization(
        self,
        model="HRP",
        codependence="pearson",
        covariance="hist",
        obj="MinRisk",
        rm="MV",
        rf=0,
        l=2,
        custom_cov=None,
        custom_mu=None,
        linkage="single",
        k=None,
        max_k=10,
        optimal_n_cluster=None,
        bins_info="KN",
        alpha_tail=0.05,
        gs_threshold=0.5,
        leaf_order=True,
        d=0.94,
        **kwargs,
    ):
        """
        Check official doc, extra parameter is optimal_n_cluster
        """
        # Covariance matrix
        if covariance == "custom_cov":
            self.cov = custom_cov.copy()
        else:
            self.cov = pe.covar_matrix(
                self.returns, method=covariance, d=0.94, **kwargs
            )

        # Custom mean vector
        if custom_mu is not None:
            if isinstance(custom_mu, pd.Series) == True:
                self.mu = custom_mu.to_frame().T
            elif isinstance(custom_mu, pd.DataFrame) == True:
                if custom_mu.shape[0] > 1 and custom_mu.shape[1] == 1:
                    self.mu = custom_mu.T
                elif custom_mu.shape[0] == 1 and custom_mu.shape[1] > 1:
                    self.mu = custom_mu
                else:
                    raise NameError("custom_mu must be a column DataFrame")
            else:
                raise NameError("custom_mu must be a column DataFrame or Series")

        self.alpha_tail = alpha_tail
        self.bins_info = bins_info
        self.gs_threshold = gs_threshold

        # Codependence matrix
        if codependence in {"pearson", "spearman", "kendall"}:
            self.codep = self.returns.corr(method=codependence).astype(float)
        elif codependence == "gerber1":
            self.codep = gs.gerber_cov_stat1(self.returns, threshold=self.gs_threshold)
            self.codep = af.cov2corr(self.codep).astype(float)
        elif codependence == "gerber2":
            self.codep = gs.gerber_cov_stat2(self.returns, threshold=self.gs_threshold)
            self.codep = af.cov2corr(self.codep).astype(float)
        elif codependence in {"abs_pearson", "abs_spearman", "abs_kendall"}:
            self.codep = np.abs(self.returns.corr(method=codependence[4:])).astype(
                float
            )
        elif codependence in {"distance"}:
            self.codep = af.dcorr_matrix(self.returns).astype(float)
        elif codependence in {"mutual_info"}:
            self.codep = af.mutual_info_matrix(self.returns, self.bins_info).astype(
                float
            )
        elif codependence in {"tail"}:
            self.codep = af.ltdi_matrix(self.returns, alpha=self.alpha_tail).astype(
                float
            )
        elif codependence in {"custom_cov"}:
            self.codep = af.cov2corr(custom_cov).astype(float)

        # Step-1: Tree clustering
        self.clusters, self.k = self._hierarchical_clustering(
            model, linkage, codependence, max_k,
            optimal_n_cluster=optimal_n_cluster, leaf_order=leaf_order
        )
        if k is not None:
            self.k = int(k)

        # Step-2: Seriation (Quasi-Diagnalization)
        self.sort_order = self._seriation(self.clusters)
        asset_order = self.assetslist
        asset_order[:] = [self.assetslist[i] for i in self.sort_order]
        self.asset_order = asset_order.copy()
        self.codep_sorted = self.codep.reindex(
            index=self.asset_order, columns=self.asset_order
        )

        # Step-2.1: Bound creation
        if self.w_max is None:
            upper_bound = pd.Series(1, index=self.asset_order)
        elif isinstance(self.w_max, pd.Series):
            upper_bound = np.minimum(1, self.w_max).loc[self.asset_order]
            if upper_bound.sum() < 1:
                raise NameError("Sum of upper bounds must be higher equal than 1")

        if self.w_min is None:
            lower_bound = pd.Series(0, index=self.asset_order)
        elif isinstance(self.w_min, pd.Series):
            lower_bound = np.maximum(0, self.w_min).loc[self.asset_order]

        if (upper_bound >= lower_bound).all().item() is False:
            raise NameError("All upper bounds must be higher than lower bounds")

        # Step-3: Recursive bisection
        if model == "HRP":
            # Recursive bisection
            weights = self._recursive_bisection(
                self.sort_order,
                rm=rm,
                rf=rf,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
        elif model in ["HERC", "HERC2"]:
            # Cluster-based Recursive bisection
            weights = self._hierarchical_recursive_bisection(
                self.clusters,
                rm=rm,
                rf=rf,
                linkage=linkage,
                model=model,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
        elif model == "NCO":
            # Step-3.1: Determine intra-cluster weights
            intra_weights = self._intra_weights(
                self.clusters, obj=obj, rm=rm, rf=rf, l=l
            )

            # Step-3.2: Determine inter-cluster weights and multiply with 􏰁→ intra-cluster weights
            weights = self._inter_weights(intra_weights, obj=obj, rm=rm, rf=rf, l=l)

        weights = weights.loc[self.asset_order]

        # Step-4: Fit weights to constraints
        if (upper_bound < weights).any().item() or (lower_bound > weights).any().item():
            max_iter = 100
            j = 0
            while (
                (upper_bound < weights).any().item()
                or (lower_bound > weights).any().item()
            ) and (j < max_iter):
                weights_original = weights.copy()
                weights = np.maximum(np.minimum(weights, upper_bound), lower_bound)
                tickers_mod = weights[
                    (weights < upper_bound) & (weights > lower_bound)
                ].index.tolist()
                weights_add = np.maximum(weights_original - upper_bound, 0).sum()
                weights_sub = np.minimum(weights_original - lower_bound, 0).sum()
                delta = weights_add + weights_sub

                if delta != 0:
                    weights[tickers_mod] += (
                        delta * weights[tickers_mod] / weights[tickers_mod].sum()
                    )

                j += 1

        weights = weights.loc[self.assetslist].to_frame()
        weights.columns = ["weights"]

        return weights
