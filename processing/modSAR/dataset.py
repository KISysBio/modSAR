"""This module represents QSAR Datasets and its operations

To create a QSARDataset, first download a list of bioactive compounds from ChEMBL:
    https://www.ebi.ac.uk/chembl/

Type in the ChEMBL ID of the desired target (e.g. CHEMBL202) and download a XLS of all
    activities.

"""

import numpy as np
import scipy.sparse as sp
import pandas as pd

from .cdk_utils import CDKUtils
from .features import apply_feature_filter


class Dataset:
    """Generic class to represent a data set"""

    def __init__(self, dataset_name, X, y, metadata=None):
        self.name = dataset_name
        self.X = X
        self.y = y
        self.metadata = metadata
        self.number_samples = X.shape[0]
        self.number_features = X.shape[1]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        object_str = "Dataset %s \n -- Samples : %d\n -- Features:  %d"
        return object_str % (self.name, self.number_samples, self.number_features)


class QSARDataset(Dataset):
    """Represents a QSAR dataset

    An object of QSARDataset is constructed from a XLS downloaded from ChEMBL

    """

    def __init__(self, name, X, y, smiles, apply_filter=True, metadata=None):
        if apply_filter:
            X = apply_feature_filter(X)
        super().__init__(name, X, y, metadata)

        cdk_utils = CDKUtils()
        self.pairwise_similarity = cdk_utils.calculate_pairwise_tanimoto(metadata, smiles)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'QSAR' + super().__repr__()
