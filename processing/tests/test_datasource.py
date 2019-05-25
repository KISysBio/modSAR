import unittest

from modSAR.datasource import ChEMBLApiDataSource


class TestChEMBLApiDataSource(unittest.TestCase):

    def test_get_activities_from_chembl(self):

        target_chembl_id = 'CHEMBL202'
        chembl_data_source = ChEMBLApiDataSource(target_id='CHEMBL202',
                                                 standard_types='IC50')
        result_df = chembl_data_source.bioactivities_df
        self.assertTrue(all(result_df['target_chembl_id'] == target_chembl_id))


class TestChEMBLFileDataSource(unittest.TestCase):
    pass


class TestGenericFileDataSource(unittest.TestCase):
    pass
