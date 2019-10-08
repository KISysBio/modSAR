"""QSAR modelling Data Sources

This module supports representation of functional bioactivies

- Directly from ChEMBL API

- From a list of Canonical SMILES, along with the measured activity

"""

import chembl_webresource_client.new_client as chembl_client
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from .utils import print_progress_bar
from .dataset import QSARDataset
from .preprocessing import Preprocessing
from .cdk_utils import CDKUtils


class DataSource(metaclass=ABCMeta):
    """
    Represent a data source

    Chemical data might be obtained from different sources:

        - ChEMBL API (class ChEMBLApiDataSource)
        - A CSV file downloaded from ChEMBL (class ChEMBLCsvDataSource)
        - A CSV file produced manually or exported from elsewhere containing two columns:
            'canonical_smiles' and 'activity' (class CsvDataSource)
          The first represent the Canonical Smiles of a compound while the second correspond
            to the acitivity of the compounds.

        DataSource parses the data and creates an object of the class modSAR.dataset.QSARDataset
          using the function `build_qsar_dataset`.

    """

    def __init__(self, target_id, smiles_column, compound_id_column, activity_column,
                 is_chembl_data, apply_filter, **kwargs):
        """
        Generic DataSource construtor

        Args:
            target_id (str): name of the protein target common to all activities in the dataset
            smiles_column (str): name of column that contains SMILES code for the compounds
            compound_id_column (str): column in the DataFrame that identifies the compound
            activity_column (str): column in the DataFrame
        """
        missing_attributes = []
        if ((smiles_column is None) or (smiles_column == '')):
            missing_attributes.append(['smiles_column'])
        if (activity_column is None):
            missing_attributes.append(['activity_column'])
        if (compound_id_column is None):
            missing_attributes.append(['compound_id_column'])
        if (is_chembl_data is None):
            missing_attributes.append(['is_chembl_data'])

        if (len(missing_attributes) > 1):
            raise ValueError('DataSource does not have required attributes: %s' %
                             (', '.join(missing_attributes)))

        self.target_id = target_id
        self.smiles_column = smiles_column
        self.compound_id_column = compound_id_column
        self.activity_column = activity_column
        self.is_chembl_data = is_chembl_data
        self.apply_filter = apply_filter
        for attr, val in kwargs.items():
            setattr(self, attr, val)

        self.bioactivities_df = self._get_bioactivities_df()

    @abstractmethod
    def _get_bioactivities_df(self):
        pass

    def build_qsar_dataset(self, calculate_similarity=True):
        """
        Preprocess bioactivities and builds a QSARDataset object
        """

        preprocess = Preprocessing(compound_id_column=self.compound_id_column,
                                   activity_column=self.activity_column,
                                   apply_chembl_filter=self.is_chembl_data,
                                   remove_duplicated=True)
        clean_df = preprocess.do(self.bioactivities_df)

        cdk_utils = CDKUtils()
        descriptors_df = cdk_utils.calculate_descriptors(clean_df, self.smiles_column)
        descriptors_df.index = clean_df.index
        X = descriptors_df

        y = clean_df[self.activity_column]
        y.index = X.index

        qsar_dataset = QSARDataset(name=self.target_id,
                                   X=X,
                                   y=y,
                                   X_smiles=clean_df[self.smiles_column],
                                   metadata=clean_df,
                                   apply_filter=self.apply_filter,
                                   calculate_similarity=calculate_similarity)
        return qsar_dataset


class ChEMBLApiDataSource(DataSource):
    """
        Retrieve active compounds for a target using ChEMBL API

        Example of use:

            chembl_data_source = ChEMBLApiDataSource(target_id='CHEMBL202', standard_types=['IC50'])
            chembl202_dataset = chembl_data_source.get_qsar_dataset()
    """

    def __init__(self, target_id, standard_types,
                 smiles_column='canonical_smiles',
                 compound_id_column='parent_molecule_chembl_id',
                 activity_column='pchembl_value',
                 apply_filter=True):
        """
        Stores a DataFrame containing the bioactivities listed on ChEMBL for specified target.
        Bioactivities can later be converted to a QSARDataset using function `get_qsar_dataset`

        Args:
            target_id (str): ChEMBL target id
            standard_types (str or list): e.g.: IC50, Ki, etc.

        """
        if type(standard_types) is str:
            standard_types = [standard_types]
        super(ChEMBLApiDataSource, self).__init__(target_id,
                                                  smiles_column=smiles_column,
                                                  compound_id_column=compound_id_column,
                                                  activity_column=activity_column,
                                                  is_chembl_data=True,
                                                  standard_types=standard_types,
                                                  apply_filter=apply_filter)

    def _get_bioactivities_df(self):
        activity = chembl_client.new_client.activity
        result = activity.filter(target_chembl_id=self.target_id,
                                 assay_type__iregex='(B|F)')

        def get_compound_df(idx, compound_dict, standard_types):
            """Filter compounds by the standard_types informed, returning a DataFrame"""

            if standard_types is None or compound_dict['standard_type'] is None:
                is_valid_std_type = False
            else:
                compound_std_type = compound_dict['standard_type'].lower()
                is_valid_std_type = any([std_type.lower() in compound_std_type
                                         for std_type in standard_types])

            if is_valid_std_type:
                # Drop unused columns
                compound_dict.pop('activity_properties', None)
                # Capture Ligand Efficiency
                lig_efficiency = compound_dict.pop('ligand_efficiency', None)

                compound_df = pd.DataFrame(compound_dict, index=[0])
                if lig_efficiency is not None:
                    lig_efficiency_df = pd.DataFrame(lig_efficiency, index=[0])
                    lig_efficiency_df.columns = ['ligand_efficiency_%s' % col
                                                 for col in lig_efficiency_df.columns]
                    compound_df = pd.concat([compound_df, lig_efficiency_df], axis=1)
            else:
                compound_df = pd.DataFrame(columns=compound_dict.keys())

            print_progress_bar(idx + 1, self.number_retrieved_compounds,
                               prefix='Progress:', suffix='Complete', length=50)
            return compound_df

        self.number_retrieved_compounds = len(result)
        print_progress_bar(0, self.number_retrieved_compounds,
                           prefix='Progress:', suffix='Complete', length=50)
        bioactivities_df = [get_compound_df(idx, compound_dict, self.standard_types)
                            for idx, compound_dict in enumerate(result)]
        bioactivities_df = pd.concat(bioactivities_df, sort=False).reset_index(drop=True)

        return bioactivities_df

    def save_bioactivities(self, xls_filename):
        self.bioactivities_df.to_excel(xls_filename, index=False)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr_str = \
            "ChEMBLApiDataSource object" \
            + "\n  target_id: " + self.target_id \
            + "\n  bioactivities: %d" % self.bioactivities_df.shape[0] \
            + "\n  standard_types: " + str(self.bioactivities_df['standard_type'].unique())
        return repr_str


class GenericFileDataSource(DataSource):
    """
        Read a file data source downloaded from ChEMBL (CSV or XLSX)

        Example of use:

            chembl_data_source = ChEMBLFileDataSource(filepath='./chembl202.xlsx', target_id='CHEMBL202')
            chembl202_dataset = chembl_data_source.get_qsar_dataset()
    """

    def __init__(self, target_id, filepath, smiles_column, compound_id_column, activity_column,
                 is_chembl_data=False, apply_filter=False):
        super().__init__(target_id,
                         smiles_column=smiles_column,
                         compound_id_column=compound_id_column,
                         activity_column=activity_column,
                         is_chembl_data=is_chembl_data,
                         filepath=filepath,
                         apply_filter=apply_filter)

    def _get_bioactivities_df(self):
        if self.filepath.endswith('.csv'):
            return pd.read_csv(self.filepath)
        else:
            return pd.read_excel(self.filepath)


class ChEMBLFileDataSource(DataSource):
    """
        Read a file data source downloaded from ChEMBL (CSV or XLSX)

        Example of use:

            chembl_data_source = ChEMBLFileDataSource(filepath='./chembl202.xlsx', target_id='CHEMBL202')
            chembl202_dataset = chembl_data_source.get_qsar_dataset()
    """

    def __init__(self, target_id, filepath,
                 smiles_column='canonical_smiles',
                 compound_id_column='parent_molecule_chembl_id',
                 activity_column='pchembl_value',
                 is_chembl_data=True,
                 apply_filter=True):
        super().__init__(target_id,
                         smiles_column=smiles_column,
                         compound_id_column=compound_id_column,
                         activity_column=activity_column,
                         is_chembl_data=is_chembl_data,
                         filepath=filepath,
                         apply_filter=apply_filter)

    def _get_bioactivities_df(self):
        if self.filepath.endswith('.csv'):
            return pd.read_csv(self.filepath)
        else:
            return pd.read_excel(self.filepath)
