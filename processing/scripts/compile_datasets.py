import xlwt
import pandas as pd

from modSAR.datasource import ChEMBLApiDataSource
from modSAR.network_algorithms import ModSAR

CHEMBL_DATASETS = ['CHEMBL245', 'CHEMBL2363', 'CHEMBL202', 'CHEMBL4777', 'CHEMBL4018']

for dataset in CHEMBL_DATASETS:
    print("Downloading CHEMBL Dataset: %s" % dataset)
    chembl_data_source = ChEMBLApiDataSource(target_id=dataset,
                                             standard_types=['Ki', 'IC50'])
    writer = pd.ExcelWriter('/mnt/data/gathered_data/%s.xls' % dataset, engine='xlwt')
    chembl_data_source.bioactivities_df.to_excel(writer, sheet_name='metadata')

    print("Calculating molecular descriptors")
    qsar_dataset = chembl_data_source.build_qsar_dataset()
    qsar_dataset.X.to_excel(writer, sheet_name='features')

    X = qsar_dataset.X
    X_norm = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    X_norm.to_excel(writer, sheet_name='normalised_features')

    qsar_dataset.Y.to_excel(writer, sheet_name='activity')
