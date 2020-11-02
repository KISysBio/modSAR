import igraph as ig
import numpy as np
import pandas as pd
import oplrareg
import rdkit

from rdkit.Chem import AllChem
from sklearn.tree import export_graphviz
from collections import Counter

from .graph import GraphUtils
from .cdk_utils import CDKUtils
from .utils import print_progress_bar
from .features import convert_to_morgan_fingerprints


class ModSAR(oplrareg.BaseOplraEstimator):
    """Implementation of ModSAR algorithm

    The current implementation relies on oplra_reg v0.2 (pip install oplra_reg) to
      create the piecewise linear regression.

    """

    def __init__(self, metadata, lam=0.01, epsilon=0.01, beta=0.03,
                 solver_name="cplex", threshold=None, k=0):
        super().__init__("modSAR", "v0.1", lam, epsilon, beta, solver_name)

        self.metadata = metadata  # Register metadata DataFrame so the algorithm can retrieve Smiles later
        self.class_names = None
        self.feature_names = None
        self.number_classes = None
        self.number_modules = None
        self.W = None
        self.B = None
        self.models = None
        self.instance_graph = None
        self.threshold = threshold
        self.k = k

    def __repr__(self):
        return super().__repr__()

    def fit(self, X, y):
        """Fits a modSAR model to the input data

        Args:
            X (pandas.DataFrame): DataFrame describing the molecular descriptors
            y (pandas.Series): The outcome variable (e.g.: pIC50)
            threshold (float): a fixed threshold to use

        """

        if X.shape[0] != len(y):
            raise ValueError("Data and target values have different dimensions.")
        # TODO: Make more checks on the input data

        smiles_col = self.metadata["Canonical_Smiles"]
        rdkit_mols = [rdkit.Chem.MolFromSmiles(smiles_col.loc[idx]) for idx in X.index]

        print("Calculating fingerprints")
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=1024)
                        for m in rdkit_mols]

        num_samples = len(X)
        total_comparisons = num_samples * (num_samples - 1)

        print("Calculating similarity")
        count = 0
        # print_progress_bar(count, total_comparisons)
        matrix = np.zeros((num_samples, num_samples), dtype="f8")
        for i in range(num_samples):
            fp1 = fingerprints[i]
            for j in range(num_samples):
                count += 1
                if j < i:
                    # print_progress_bar(count, total_comparisons)
                    continue
                elif j == i:
                    matrix[i, j] = 0
                else:
                    fp2 = fingerprints[j]

                    try:
                        sim = rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)
                    except Exception as e:
                        error_msg = "Error calculating similarity: %s x %s\n%s"
                        warnings.warn(error_msg % (df_smiles.index[i], df_smiles.index[j], e))
                        sim = np.nan
                    matrix[i, j] = sim
                    matrix[j, i] = sim
                # print_progress_bar(count, total_comparisons)

        # Convert numpy matrix to pandas DataFrame
        similarity_df = pd.DataFrame(matrix, index=X.index, columns=X.index)

        if self.k != 0:
            is_weighted = True
        else:
            is_weighted = False

        # Create graph with appropriate threshold (if not predefined)
        if self.threshold is None:
            g, threshold = GraphUtils.find_optimal_threshold(similarity_df)
        else:
            g = GraphUtils.create_graph(similarity_df, threshold, k,
                                        is_directed=False, is_weighted=is_weighted)

        self.threshold = threshold
        print("Threshold: {} | k: {}".format(threshold, self.k))

        communities = np.array(g.vs["louvain"], dtype="<U8")
        counter_comm = Counter(communities)
        self.number_modules = len(counter_comm.keys())
        print("Communities: %s" % counter_comm)
        g.vs['label'] = X.index
        g.vs["community"] = communities
        self.instance_graph = g
        self.class_names = list(set(communities))
        self.feature_names = X.columns

        self.models = {}
        self.W = {}
        self.B = {}

        for comm, count in counter_comm.items():
            print("Num. samples in comm %s: %d" % (comm, count))
            samplesInCommunity = np.argwhere(communities == comm).transpose()[0]
            currentX = X.iloc[samplesInCommunity].reset_index(drop=True)
            currentY = y.iloc[samplesInCommunity, 0].reset_index(drop=True)

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
        self.fingerprints_training = fingerprints

    def classify_sample(self, fp_sample):
        """"""

        similarities = [rdkit.DataStructs.TanimotoSimilarity(fp_sample, fp_training)
                        for fp_training in self.fingerprints_training]
        similarities = np.array(similarities)

        neighbours = np.argwhere(similarities >= self.threshold).flatten()
        if len(neighbours) <= 1:
            neighbours = np.array([similarities.argmax()])

        modules_training = np.array(self.instance_graph.vs["community"])
        neighbours_modules = Counter(modules_training[neighbours])
        # Maximum number of links to a module
        max_links = neighbours_modules.most_common(1)[0][1]
        max_modules = [k for k, v in neighbours_modules.items() if v == max_links]

        if len(max_modules) == 1:
            closest_module = max_modules[0]
        else:
            dist_to_modules = {comm: np.average(similarities[np.argwhere(modules_training == comm).flatten()])
                               for comm in max_modules}
            closest_module = max(dist_to_modules, key=lambda x: dist_to_modules[x])
        return closest_module

    def predict(self, X):

        X_fingerprints = convert_to_morgan_fingerprints(X)

        modules = np.array([self.classify_sample(fingerprint) for fingerprint in X_fingerprints])

        counter_modules = Counter(modules)
        final_predictions = np.zeros(X.shape[0])

        for comm, count in counter_modules.items():
            samples_in_community = np.argwhere(modules == comm).transpose()[0]

            if(all(self.models[comm].final_model.get_coefficients().flatten() == 0)):
                final_predictions[samples_in_community] = self.models[comm].final_model.get_intercepts().flatten()[0]
            else:
                if type(X) == pd.DataFrame:
                    X_samples = X.iloc[samples_in_community]
                else:
                    X_samples = X[samples_in_community, ]
                final_predictions[samples_in_community] = self.models[comm].predict(X_samples)

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
        coefficients = coefficients.loc[:, newColumns]
        breakpoints = breakpoints.loc[:, ['module', 'region', 'breakpoints', 'fStar']]

        return coefficients, breakpoints
