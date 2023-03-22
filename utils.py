import numpy as np


def _nearest_metric(u, v, s=None, dists=None, clusters=None):
    if dists is None:
        pairwise_dst = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
        return pairwise_dst.min()

    alpha_u = 1 / 2
    alpha_v = 1 / 2
    beta = 0
    gamma = -1 / 2

    return alpha_u * dists[u, s] + \
           alpha_v * dists[v, s] + \
           beta * dists[u, v] + \
           gamma * np.abs(dists[u, s] - dists[v, s])


def _farthest_metric(u, v, s=None, dists=None, clusters=None):
    if dists is None:
        pairwise_dst = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
        return pairwise_dst.max()

    alpha_u = 1 / 2
    alpha_v = 1 / 2
    gamma = 1 / 2

    return alpha_u * dists[u, s] + \
           alpha_v * dists[v, s] + \
           gamma * np.abs(dists[u, s] - dists[v, s])


def _group_mean_metric(u, v, s=None, dists=None, clusters=None):
    if dists is None:
        pairwise_dst = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
        return pairwise_dst.sum() / len(u) / len(v)

    alpha_u = 0
    alpha_v = 0

    return alpha_u * dists[u, s] + \
           alpha_v * dists[v, s]


def _center_mean_metric(u, v, s=None, dists=None, clusters=None):
    if dists is None:
        center1 = u.mean(0)
        center2 = v.mean(0)
        return np.linalg.norm(center1 - center2)

    alpha_u = 0
    alpha_v = 0
    beta = -alpha_u * alpha_v

    return alpha_u * dists[u, s] + \
           alpha_v * dists[v, s] + \
           beta * dists[u, v]


def _ward_metric(u, v, s=None, dists=None, clusters=None):
    if dists is None:
        center1 = u.mean(0)
        center2 = v.mean(0)
        return len(u) * len(v) * np.linalg.norm(center1 - center2) / (
                    len(u) + len(v))
    alpha_u = (len(clusters[s]) + len(clusters[u])) / \
              (len(clusters[s]) + len(clusters[u]) + len(clusters[v]))

    alpha_v = (len(clusters[s]) + len(clusters[v])) / \
              (len(clusters[s]) + len(clusters[u]) + len(clusters[v]))

    beta = -len(clusters[s]) / \
           (len(clusters[s]) + len(clusters[u]) + len(clusters[v]))

    return 1


class LanceWilliamsClustering:
    def __init__(self, metric):
        self.distances = None
        self.clusters = None
        self.id2point = None
        name2metric = {'nearest': _nearest_metric,
                       'farthest': _farthest_metric,
                       'group_mean': _group_mean_metric,
                       'center': _center_mean_metric,
                       'ward': _ward_metric}
        assert metric in name2metric
        self.metric = name2metric[metric]

    def clusterize(self, points, num_iters):
        self.id2point = {i: points[i, :] for i in range(len(points))}
        self.clusters = [[i] for i in range(len(points))]
        self.distances = np.zeros((len(points), len(points)))

        for i in range(len(points)):
            for j in range(len(points)):
                self.distances[i, j] = self.metric(points[i:i+1, :],
                                                   points[j:j+1, :])

        np.fill_diagonal(self.distances, np.max(self.distances) + 10)

        for _ in range(num_iters):
            u, v = np.unravel_index(np.argmin(self.distances),
                                    self.distances.shape)
            new_metrics = np.array([self.metric(u, v, s,
                                                self.distances,
                                                self.clusters)
                                    for s in range(len(self.clusters))])
            new_cluster = self.clusters[u] + self.clusters[v]
            del self.clusters[max(v, u)]
            del self.clusters[min(v, u)]
            self.clusters.append(new_cluster)
            assert sum([len(x) for x in self.clusters]) == len(points)
            self.distances = np.append(self.distances, np.array([new_metrics]), axis=0)
            new_metrics = np.append(new_metrics, np.max(self.distances))
            self.distances = np.append(self.distances, np.array([new_metrics]).T, axis=1)
            self.distances = np.delete(self.distances, max(v, u), axis=0)
            self.distances = np.delete(self.distances, min(v, u), axis=0)

            self.distances = np.delete(self.distances, max(v, u), axis=1)
            self.distances = np.delete(self.distances, min(v, u), axis=1)
            np.fill_diagonal(self.distances, np.max(self.distances))

        return self.clusters
