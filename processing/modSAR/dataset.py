"""This module represents QSAR Datasets and its operations

To create a QSARDataset, first download a list of bioactive compounds from ChEMBL:
    https://www.ebi.ac.uk/chembl/

Type in the ChEMBL ID of the desired target (e.g. CHEMBL202) and download a XLS of all
    activities.

"""

import xlwt
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

    def __init__(self, name, X, y, X_smiles, apply_filter=True, metadata=None):
        if apply_filter:
            X = apply_feature_filter(X)
        super().__init__(name, X, y, metadata)

        self.y.index = X.index
        X_smiles.index = X.index
        self.X_smiles = X_smiles
        self.pairwise_similarity = CDKUtils().calculate_pairwise_tanimoto(X_smiles)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'QSAR' + super().__repr__()


class QSARDatasetIO():
    """Parses an Excel file into a QSARDataset object"""

    @staticmethod
    def load(filepath, dataset_name,
             features_sheetname='normalised_features',
             metadata_sheetname='metadata',
             smiles_column='CANONICAL_SMILES',
             id_column='PARENT_CMPD_CHEMBLID',
             activity_sheetname='pchembl_value',
             apply_filter=False):
        X = pd.read_excel(filepath, sheet_name=features_sheetname)
        y = pd.read_excel(filepath, sheet_name=activity_sheetname)

        metadata = pd.read_excel(filepath, sheet_name=metadata_sheetname)
        X.index = metadata[id_column].values
        y.index = X.index
        return QSARDataset(dataset_name, X, y, X_smiles=metadata[smiles_column],
                           apply_filter=apply_filter, metadata=metadata)

    @staticmethod
    def write(qsar_dataset, filepath,
              features_sheetname='normalised_features',
              metadata_sheetname='metadata',
              activity_sheetname='activity'):

        with pd.ExcelWriter(filepath, engine='xlwt') as writer:
            qsar_dataset.X.to_excel(writer, sheet_name=features_sheetname, index=False)
            qsar_dataset.y.to_excel(writer, sheet_name=activity_sheetname, index=False)
            qsar_dataset.metadata.to_excel(writer, sheet_name=metadata_sheetname, index=False)
