import numpy as np
import igraph as ig


class GraphUtils:

    @staticmethod
    def create_graph(similarity_matrix, threshold, k=5, is_directed=True, is_weighted=True):
        """Apply a threshold to adjacency matrix to return a sparser graph

        Args:
            similarity_matrix (np.array): n x n matrix
        """

        if type(similarity_matrix) != np.ndarray:
            raise ValueError('Similarity matrix is not a numpy array. Type = %s' % type(similarity_matrix))
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError('Similarity matrix is not symmetrical. Dim: %d x %d' % (similarity_matrix.shape[0], similarity_matrix.shape[1]))
        if (similarity_matrix < 0).any() or (similarity_matrix > 1).any():
            raise ValueError('Similarity matrix is invalid. It contains values beyond range [0, 1]')
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold beyond range [0, 1]. Threshold = %.4f' % threshold)

        thresholdMat = (similarity_matrix >= threshold) * similarity_matrix
        thresholdMat = thresholdMat.tolist()

        g = ig.Graph.Weighted_Adjacency(thresholdMat, loops=False)

        if not is_directed:
            if is_weighted:
                w = g.es["weight"]
                g.to_undirected()
                g.es["weight"] = w
            else:
                g.to_undirected()

        belowThreshold = [i for i, s in enumerate(g.strength(mode=1)) if s < k]
        edgeId = len(g.es())

        for n in belowThreshold:
            neighbours = np.flip(np.argsort(similarity_matrix.iloc[n].values), axis=0)[0:k]
            for neigh in neighbours:
                shouldAdd = not g.es.select(_source=n, _target=neigh) and (is_directed or (not is_directed and not g.es.select(_source=neigh, _target=n)))
                if shouldAdd:
                    g.add_edge(n, neigh)
                    if is_weighted:
                        g.es[edgeId]["weight"] = similarity_matrix.iloc[n, neigh]
                    edgeId += 1
            g.vs[n]["usedKnn"] = True

        g.vs["degree"] = g.degree()

        if is_weighted:
            g["globalClusteringCoefficient"] = g.transitivity_avglocal_undirected(weights="weight", mode="zero")
        else:
            g["globalClusteringCoefficient"] = g.transitivity_avglocal_undirected(mode="zero")

        if is_weighted:
            vc = g.community_multilevel(weights='weight')
        else:
            vc = g.community_multilevel()
        g.vs["louvain"] = ['m%02d' % (x + 1) for x in vc.membership]
        g["threshold"] = threshold
        g["k"] = k

        # g["edgeDensity"] = g.density()
        # g["metric"] = g["globalClusteringCoefficient"] - 0.4 * g["edgeDensity"]
        return g
