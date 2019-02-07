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
from .datasource import ChEMBLApiDataSource
from .cdk_utils import CDKDescriptors


class Preprocessing:
    """Preprocess data obtained via a DataSource (.csv file or ChEMBL)

    The bioactivity data is filtered, duplicated and invalid entries are handled and
      molecular descriptors are calculated.
    """

    def __init__(self, activity_column='median_activity_value'):
        self.activity_column = activity_column

    def _mark_to_remove(self, data, activity_column):
        """Mark data for removal if standard deviation of duplicated entries is above 1"""

        activity_value = data[activity_column].astype(float)
        std_deviation = np.std(pchembl_value)
        number_compounds = data.shape[0]

        # It is duplicated
        result_df = pd.Series(
            {'number_compounds': number_compounds,
             'standard_deviation': std_deviation,
             self.activity_column: np.median(activity_value),
             'duplicated': number_compounds > 1,
             'mark_to_remove': std_deviation > 1})
        return result_df

    def _default_filter_chembl(self, bioactivities_df):
        """Default filter for data downloaded from ChEMBL"""

        # Relation measured must be of type equality
        valid_rows = bioactivities_df['relation'] == '='
        bioactivities_df = bioactivities_df[valid_rows]

        # Remove data that is marked as potential invalid
        valid_rows = bioactivities_df['data_validity_comment'].isnull()
        bioactivities_df = bioactivities_df[valid_rows]

        # There must be a valid pCHEMBL value
        valid_rows = ~bioactivities_df['pchembl_value'].isnull()
        bioactivities_df = bioactivities_df[valid_rows]
        return bioactivities_df

    def clean_dataset(self, data_source):

        bioactivities_df = data_source.bioactivities_df.copy()

        if self.data_source_type is ChEMBLApiDataSource:
            ## Perform initial filter on bioactivities
            bioactivities_df = self._default_filter_chembl(bioactivities_df)
        else:
            raise ValueError("Data source not recognized: %s" % type(data_source))

        # Mark duplicated entries for removeal
        grouped_dataset = bioactivities_df.groupby(data_source.compound_id_column)
        grouped_dataset = grouped_dataset.apply(lambda x: _mark_to_remove(x, data_source.activity_column))

        # The resulting dataset, clean_df, does not contain duplicated entries
        merged_df = pd.merge(bioactivities_df, grouped_dataset.reset_index())
        merged_df = merged_df[~merged_df['mark_to_remove']]
        clean_df = merged_df.groupby(data_source.compound_id).head(1)

        return clean_df


def build_qsar_dataset(data_source):
    # TODO: Complex this flow, make sure it works

    clean_df = Preprocessing().clean_dataset(data_source)
    descriptors = CDKDescriptors.calculate(clean_df, data_source.smiles_column)
    qsar_dataset = QSARDataset(descriptors, activity=clean_dfmetadata=clean_df)


class Dataset:
    """Generic class to represent a data set"""

    def __init__(self, dataset_name, data, activity_column):


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        object_str = "Dataset %s \n -- Samples : %d\n -- Features:  %d"
        return object_str % (self.dataset_name, self.number_samples, self.number_features)


class QSARDataset(Dataset):
    """Represents a QSAR dataset

    An object of QSARDataset is constructed from a XLS downloaded from ChEMBL

    """

    def __init__(self, target_id, bioactivities_df,
                 activity_column='median_pchembl_value'):



        self.target_id = target_id
        # self.metadata.set_index("ID", inplace=True)

        self.features = descriptors
        # self.features.set_index(self.metadata["ID"])

        # self.normalised_features = excelFile.parse("normalised_features")
        # self.normalised_features.set_index(self.metadata["ID"], inplace=True)

        # self.fingerprint = self._parse_fingerprint(excelFile.parse("ECFP4_bits").set_index(self.metadata["ID"])["BITS"])
        super().__init__(self.target_id, self.features, activity_column)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'QSAR' + super.__repr__()
