import igraph as ig
import numpy as np
import pandas as pd
import oplrareg

from sklearn.tree import export_graphviz
from collections import Counter

from .graph import GraphUtils
from .cdk_utils import CDKUtils


class ModSAR(oplrareg.BaseOplraEstimator):
    """Implementation of ModSAR algorithm

    The current implementation relies on oplra_reg v0.1 (pip install oplra_reg) to
      create the piecewise linear regression.

    """

    def __init__(self, lam=0.01, epsilon=0.01, beta=0.03, solver_name="cplex"):
        super().__init__("modSAR", "v0.1", lam, epsilon, beta, solver_name)

        self.class_names = None
        self.feature_names = None
        self.number_classes = None
        self.number_modules = None
        self.W = None
        self.B = None
        self.models = None
        self.instance_graph = None

    def fit(self, X, y, similarity_matrix, X_smiles, threshold=None, k=0):
        """Fits a modSAR model to the input data

        Args:
            X (pandas.DataFrame): DataFrame describing the molecular descriptors
            y (pandas.Series): The outcome variable (e.g.: pIC50)
            threshold (float): a fixed threshold to use

        """

        if X.shape[0] != len(y):
            raise ValueError("Data and target values have different dimensions.")
        # TODO: Make more checks on the input data

        print(self)

        if k != 0:
            is_weighted = True
        else:
            is_weighted = False

        # Create graph with appropriate threshold (if not predefined)
        if threshold is None:
            # Find best threshold
            bestThreshold = None
            bestClusteringCoefficient = None
            bestG = None
            for threshold in np.linspace(0.20, 0.40, 20):
                g = GraphUtils.create_graph(similarity_matrix, threshold, k, is_directed=False, is_weighted=is_weighted)
                if bestThreshold is None or g["globalClusteringCoefficient"] > bestClusteringCoefficient:
                    bestG = g
                    bestThreshold = threshold
                    bestClusteringCoefficient = g["globalClusteringCoefficient"]
            g = bestG
            threshold = bestThreshold
            print("Best Threshold = %.2f | ACC = %.3f" % (bestThreshold, bestClusteringCoefficient))
        else:
            g = create_graph(similarity_matrix, threshold, k, is_directed=False, is_weighted=is_weighted)

        self.threshold = threshold
        self.k = k
        print("Threshold: {} | k: {}".format(threshold, k))

        self.models = {}
        self.W = {}
        self.B = {}

        communities = np.array(g.vs["louvain"], dtype="<U8")
        counter_comm = Counter(communities)
        self.number_modules = len(counter_comm.keys())

        for comm, count in counter_comm.items():
            print("Num. samples in comm %s: %d" % (comm, count))
            samplesInCommunity = np.argwhere(communities == comm).transpose()[0]
            currentX = X.iloc[samplesInCommunity].reset_index(drop=True)
            currentY = y[samplesInCommunity].reset_index(drop=True)

            oplra_regularised = oplrareg.OplraRegularised(self.lam,
                                                          self.epsilon,
                                                          beta=self.beta,
                                                          solver_name=self.solver_name)

            oplra_regularised.fit(currentX, currentY)
            self.models[comm] = oplra_regularised

            for region in range(oplra_regularised.final_model.number_regions):
                self.W["%s-r%d" % (comm, region)] = {}
                for f in oplra_regularised.final_model.f:
                    self.W["%s-r%d" % (comm, region)].update({f: oplra_regularised.final_model.W[region, f].value})
                self.B["%s-r%d" % (comm, region)] = oplra_regularised.final_model.B[region].value

        self.number_classes = len(self.W.keys())

        counter_comm = Counter(communities)
        print("Communities: %s" % counter_comm)
        g.vs["community"] = communities
        g.vs['SMILES'] = X_smiles
        self.instance_graph = g
        self.class_names = list(set(communities))
        self.feature_names = X.columns

    def classify_samples(self, g, similarityMatrix):
        """
        Classify samples according to their neighbourhood in the graph

        Args:
            g: graph obtained during training (fit)
            similarityMatrix: n x m matrix of similarity between the n samples to be classified and m nodes in graph g
            param majority: should a sample be assigned to the module where most of its connections are? (majority = True) *default
                                or should it be assigned to the module where the average similarity is higher? (majority = False)
        """

        samples = range(similarityMatrix.shape[0])
        thresholdMat = (similarityMatrix >= self.threshold) * similarityMatrix
        noNeighbours = thresholdMat.astype(bool).sum(axis=1)
        instanceModules = np.array(g.vs["community"])
        classes = np.empty(len(samples), dtype="<U8")
        for s in samples:
            if noNeighbours[s] <= 1:
                if self.k == 0:
                    neighbours = similarityMatrix.iloc[s, :].values.argmax()
                else:
                    neighbours = similarityMatrix.iloc[s, :].values.argsort()[(len(samples) - self.k):]
            else:
                neighbours = np.argwhere(thresholdMat.iloc[s, :] > 0).flatten()

            counterNeigh = Counter(instanceModules[neighbours])

            # Maximum number of links to a module
            maxLinks = counterNeigh.most_common(1)[0][1]
            maxModules = [k for k, v in counterNeigh.items() if v == maxLinks]

            if len(maxModules) == 1:
                classes[s] = maxModules[0]
            else:
                modulesDist = {comm: np.average(similarityMatrix.iloc[s, np.argwhere(instanceModules == comm).flatten()])
                               for comm in maxModules}
                classes[s] = max(modulesDist, key=lambda x: modulesDist[x])
        return classes

    def predict(self, X, X_smiles):

        similarity_matrix = get_asym_similarity_matrix(fingerprints, self.fingerprints)

        classes = self.classify_samples(self.instance_graph, similarity_matrix)

        counter_classes = Counter(classes)
        final_predictions = np.zeros(X.shape[0])

        for comm, count in counter_classes.items():
            samples_in_community = np.argwhere(classes == comm).transpose()[0]
            final_predictions[samples_in_community] = self.models[comm].predict(X.iloc[samples_in_community])

        return final_predictions

    def get_model_info(self):
        coefficients = pd.DataFrame()
        breakpoints = pd.DataFrame()
        for comm, m in self.models.items():
            coeffs, bkpoints = m.get_model_info()
            coeffs.insert(0, 'module', comm)
            bkpoints.insert(0, 'module', comm)
            bkpoints.insert(1, 'region', range(bkpoints.shape[0]))
            coefficients = coefficients.append(coeffs, ignore_index=True)
            breakpoints = breakpoints.append(bkpoints, ignore_index=True)

        newColumns = ['module', 'region'] + list([a for a in coefficients.columns if a not in ['module', 'region', 'B']]) + ['B']
        coefficients = coefficients.reindex_axis(newColumns, axis=1)
        breakpoints = breakpoints.reindex_axis(['module', 'region', 'breakpoints', 'fStar'], axis=1)

        return coefficients, breakpoints
