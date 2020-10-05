from typing import Optional, Dict

import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift, AffinityPropagation, Birch, OPTICS, DBSCAN, \
    AgglomerativeClustering

from average_image.utils import denormalize, pca_cluster_center


class Clustering(object):
    def __init__(self, data):
        self.data = data
        self.algo = None
        self.cluster_centers = None
        self.labels = None
        self.cluster_distribution = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        return NotImplemented

    def _renormalize(self, options):
        cc_0 = denormalize(self.cluster_centers[..., 0], options['max_1'], options['min_1'])
        cc_1 = denormalize(self.cluster_centers[..., 1], options['max_0'], options['min_0'])
        self.cluster_centers[..., 0] = cc_0
        self.cluster_centers[..., 1] = cc_1

    @staticmethod
    def renormalize_any_cluster(cluster_centers, options):
        cc_0 = denormalize(cluster_centers[..., 0], options['max_1'], options['min_1'])
        cc_1 = denormalize(cluster_centers[..., 1], options['max_0'], options['min_0'])
        cluster_centers[..., 0] = cc_0
        cluster_centers[..., 1] = cc_1
        return cluster_centers


class MeanShiftClustering(Clustering):
    def __init__(self, data, bandwidth: Optional[float] = None, quantile: float = 0.1, n_jobs: int = 8,
                 bin_seeding: bool = False, cluster_all: bool = True, max_iter: int = 300, min_bin_freq=3):
        super(MeanShiftClustering, self).__init__(data=data)
        if bandwidth is None:
            bandwidth = estimate_bandwidth(self.data, quantile=quantile, n_jobs=n_jobs)
        self.bandwidth = bandwidth
        self.algo = MeanShift(bandwidth=self.bandwidth, bin_seeding=bin_seeding, cluster_all=cluster_all,
                              max_iter=max_iter, min_bin_freq=min_bin_freq)

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.cluster_centers = self.algo.cluster_centers_

        if renormalize:
            self._renormalize(options)

    def cluster_pca(self):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.cluster_centers = self.algo.cluster_centers_
        return pca_cluster_center(self.cluster_centers)


class AffinityPropagationClustering(Clustering):
    def __init__(self, data, preference: Optional[int] = None, damping: float = 0.5, max_iter: int = 200,
                 convergence_iter: int = 15,
                 verbose: bool = False):
        super(AffinityPropagationClustering, self).__init__(data=data)
        self.algo = AffinityPropagation(preference=preference, damping=damping, max_iter=max_iter, verbose=verbose,
                                        convergence_iter=convergence_iter)
        self.cluster_centers_indices = None
        self.affinity_matrix = None
        self.n_iter = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.cluster_centers = self.algo.cluster_centers_
        self.cluster_centers_indices = self.algo.cluster_centers_indices_
        self.affinity_matrix = self.algo.affinity_matrix_
        self.n_iter = self.algo.n_iter_

        if renormalize:
            self._renormalize(options)


class OPTICSClustering(Clustering):
    def __init__(self, data, min_samples: int = 5, max_eps: float = np.inf, metric: str = 'minkowski', p: int = 2,
                 metric_params: dict = None, cluster_method: str = 'xi', eps: Optional[float] = None, xi: float = 0.05,
                 predecessor_correction: bool = True, min_cluster_size: Optional[float] = None, algorithm: str = 'auto',
                 leaf_size: int = 30, n_jobs: int = None):
        super(OPTICSClustering, self).__init__(data=data)
        self.algo = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, p=p, metric_params=metric_params,
                           cluster_method=cluster_method, eps=eps, xi=xi, predecessor_correction=predecessor_correction,
                           min_cluster_size=min_cluster_size, algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs)
        self.reachability = None
        self.ordering = None
        self.core_distances = None
        self.predecessor = None
        self.cluster_hierarchy = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.reachability = self.algo.reachability_
        self.ordering = self.algo.ordering_
        self.core_distances = self.algo.core_distances_
        self.predecessor = self.algo.predecessor_
        self.cluster_hierarchy = self.algo.cluster_hierarchy_

        if renormalize:
            self._renormalize(options)


class DBSCANClustering(Clustering):
    def __init__(self, data, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean',
                 metric_params: dict = None, algorithm: str = 'auto', leaf_size: int = 30, p: float = None,
                 n_jobs: int = None):
        super(DBSCANClustering, self).__init__(data=data)
        self.algo = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                           algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        self.core_sample_indices = None
        self.components = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.core_sample_indices = self.algo.core_sample_indices_
        self.components = self.algo.components_

        # if renormalize:
        #     self._renormalize(options)


class HierarchicalClustering(Clustering):
    def __init__(self, data, n_clusters: Optional[int] = 2, affinity: str = 'euclidean', memory: Optional[str] = None,
                 connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold: float = None):
        super(HierarchicalClustering, self).__init__(data=data)
        self.algo = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, memory=memory,
                                            connectivity=connectivity, compute_full_tree=compute_full_tree,
                                            linkage=linkage, distance_threshold=distance_threshold)
        self.n_clusters = None
        self.n_leaves = None
        self.n_connected_components = None
        self.children = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.n_clusters = self.algo.n_clusters_
        self.n_leaves = self.algo.n_leaves_
        self.n_connected_components = self.algo.n_connected_components_
        self.children = self.algo.children_

        if renormalize:
            self._renormalize(options)


class BirchClustering(Clustering):
    def __init__(self, data, threshold: float = 0.5, branching_factor: int = 50, n_clusters: int = 3,
                 compute_labels: bool = True):
        super(BirchClustering, self).__init__(data=data)
        self.algo = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters,
                          compute_labels=compute_labels)
        self.root = None
        self.dummy_leaf = None
        self.sub_cluster_centers = None
        self.sub_cluster_labels = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.root = self.algo.root_
        self.dummy_leaf = self.algo.dummy_leaf_
        self.sub_cluster_centers = self.algo.subcluster_centers_
        self.sub_cluster_labels = self.algo.subcluster_labels_

        if renormalize:
            self._renormalize(options)
