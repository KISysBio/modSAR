"""This module represents QSAR Datasets and its operations

To create a QSARDataset, first download a list of bioactive compounds from ChEMBL:
    https://www.ebi.ac.uk/chembl/

Type in the ChEMBL ID of the desired target (e.g. CHEMBL202) and download a XLS of all
    activities.

"""

import numpy as np
import scipy.sparse as sp
import pandas as pd

from rdkit.DataStructs import cDataStructs


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

    def __init__(self, name, X, y, filter_invalid=True, metadata=None):
        super().__init__(name, X, y, metadata)

        # self.fingerprint = self._parse_fingerprint(excelFile.parse("ECFP4_bits").set_index(self.metadata["ID"])["BITS"])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'QSAR' + super().__repr__()
