import os
import pandas as pd

from modSAR.dataset import QSARDatasetIO

from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import PredefinedSplit, ParameterGrid, ParameterSampler


class DataSplit:

    def __init__(self, qsar_dataset, n_splits=100):
        self.n_splits = n_splits
        self.qsar_dataset = qsar_dataset
        filename = "/mnt/data/%s_splits.xlsx" % (self.qsar_dataset.name)
        excelFile = pd.ExcelFile(filename)
        self.sheets = {sheetName: excelFile.parse(sheetName) for sheetName in excelFile.sheet_names}

    # Get ID of samples:
    def get_id_internal_samples(self, split_number):
        return self.sheets["split%d_internal_samples" % split_number]["ID"]

    def get_id_external_samples(self, split_number):
        return self.sheets["split%d_external_samples" % split_number]["ID"]

    def get_id_internal_ts_samples(self, split_number, fold_number):
        folds = self.sheets["split%d_folds" % split_number]
        return folds.loc[folds["fold"] == fold_number, "ID"]

    def get_id_internal_tr_samples(self, split_number, fold_number):
        internal_samples = self.get_internal_samples(split_number)
        ts_samples = self.get_id_internal_ts_samples(split_number, fold_number)
        return internal_samples[~internal_samples.index.isin(ts_samples)].index

    # Subset according to predefined split and fold number:
    def get_internal_samples(self, split_number):
        return self.qsar_dataset.X.loc[self.get_id_internal_samples(split_number)]

    def get_external_samples(self, split_number):
        return self.qsar_dataset.X.loc[self.get_id_external_samples(split_number)]

    def get_test_samples_in_fold(self, split_number, fold_number):
        return self.qsar_dataset.X.loc[self.get_id_internal_ts_samples(split_number, fold_number)]

    def get_train_samples_in_fold(self, split_number, fold_number):
        return self.qsar_dataset.X.loc[self.get_id_internal_tr_samples(split_number, fold_number)]

    def get_internal_Y(self, split_number):
        return self.qsar_dataset.y.loc[self.get_id_internal_samples(split_number)]

    def get_external_Y(self, split_number):
        return self.qsar_dataset.y.loc[self.get_id_external_samples(split_number)]

    def get_test_Y_in_fold(self, split_number, fold_number):
        return self.qsar_dataset.y.loc[self.get_id_internal_ts_samples(split_number, fold_number)]

    def get_train_Y_in_fold(self, split_number, fold_number):
        return self.qsar_dataset.y.loc[self.get_id_internal_tr_samples(split_number, fold_number)]


class QSARValidation:
    """ Performs QSAR validation workflow
    """

    def __init__(self, estimator, data_split, split_number, is_random_search=False):
        self.estimator = estimator
        self.is_random_search = is_random_search
        self.data_split = data_split
        self.split_number = split_number
