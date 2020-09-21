from typing import Optional, Dict

from sklearn.cluster import estimate_bandwidth, MeanShift

from utils import denormalize


class MeanShiftClustering(object):
    def __init__(self, data, bandwidth: Optional[float] = None, quantile: float = 0.1, n_jobs: int = 8,
                 bin_seeding: bool = False, cluster_all: bool = True, max_iter: int = 300, min_bin_freq=3):
        super(MeanShiftClustering, self).__init__()
        self.data = data
        if bandwidth is None:
            bandwidth = estimate_bandwidth(self.data, quantile=quantile, n_jobs=n_jobs)
        self.bandwidth = bandwidth
        self.algo = MeanShift(bandwidth=self.bandwidth, bin_seeding=bin_seeding, cluster_all=cluster_all,
                              max_iter=max_iter, min_bin_freq=min_bin_freq)
        self.labels = None
        self.cluster_centers = None
        self.cluster_distribution = None

    def cluster(self, renormalize: bool = False, options: Optional[Dict] = None):
        self.algo.fit(self.data)
        self.labels = self.algo.labels_
        self.cluster_centers = self.algo.cluster_centers_

        if renormalize:
            # maybe for portrait images we need to switch?
            cc_0 = denormalize(self.cluster_centers[..., 0], options['max_1'], options['min_1'])
            cc_1 = denormalize(self.cluster_centers[..., 1], options['max_0'], options['min_0'])
            self.cluster_centers[..., 0] = cc_0
            self.cluster_centers[..., 1] = cc_1
