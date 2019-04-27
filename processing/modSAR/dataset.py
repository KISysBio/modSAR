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

    def __init__(self, dataset_name, data, activity_column):
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        object_str = "Dataset %s \n -- Samples : %d\n -- Features:  %d"
        return object_str % (self.dataset_name, self.number_samples, self.number_features)


class QSARDataset(Dataset):
    """Represents a QSAR dataset

    An object of QSARDataset is constructed from a XLS downloaded from ChEMBL

    """

    def __init__(self, descriptors_df, activity_values, target_id, metadata=None):
        self.descriptors_df = descriptors_df
        self.activity = activity
        self.target_id = target_id
        self.features = descriptors_df
        self.metadata = metadata

        # self.fingerprint = self._parse_fingerprint(excelFile.parse("ECFP4_bits").set_index(self.metadata["ID"])["BITS"])
        # super().__init__(self.target_id, self.features, activity_column)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'QSAR' + super.__repr__()
