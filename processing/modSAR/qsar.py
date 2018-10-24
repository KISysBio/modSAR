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

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.number_samples = data.shape[0]
        self.number_features = data.shape[1]
        self.feature_names = data.dtypes.index.tolist()


class QSARDataset(Dataset):
    """Represents a QSAR dataset

    An object of QSARDataset is constructed from a XLS downloaded from ChEMBL

    """

    def __init__(self, xls_file, dataset_name):
        self.name = dataset_name
        excelFile = pd.ExcelFile(xls_file)

        self.metadata = excelFile.parse("metadata")
        self.metadata.set_index("ID", inplace=True)

        self.features = excelFile.parse("features")
        self.features.set_index(self.metadata["ID"])

        self.normalised_features = excelFile.parse("normalised_features")
        self.normalised_features.set_index(self.metadata["ID"], inplace=True)

        # self.fingerprint = self._parse_fingerprint(excelFile.parse("ECFP4_bits").set_index(self.metadata["ID"])["BITS"])
        super().__init__(self.normalised_features, self.metadata["NEW_ACTIVITY_VALUE"])

    def _parse_fingerprint(self, ecfp4Bits):
        activeBits = ecfp4Bits.apply(lambda bits: np.array(list(map(int, bits.split("|")))) - 1)
        fpMatrix = sp.lil_matrix((activeBits.shape[0], 1024), dtype=np.int16)
        for i in range(activeBits.shape[0]):
            fpMatrix[i, activeBits[i]] = 1
        fpDF = pd.DataFrame({'FINGERPRINT': np.apply_along_axis(lambda x: "".join(x.astype("str")), 1, fpMatrix.toarray())},
                            index=self.metadata["ID"])
        fpDF["FP_OBJ"] = [cDataStructs.CreateFromBitString(bits) for bits in fpDF["FINGERPRINT"]]
        return fpDF

    def __str__(self):
        object_str = "Dataset %s \n -- Samples : %d\n -- Features:  %d"
        return object_str % (self.name, self.number_samples, self.number_features)
