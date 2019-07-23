import os
import time
import numpy as np
import pandas as pd

from oplrareg.solvers import get_solver_definition
from modSAR.dataset import QSARDatasetIO
from modSAR.graph import GraphUtils

from copy import deepcopy
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import PredefinedSplit, ParameterGrid, ParameterSampler


class DataSplit:

    def __init__(self, qsar_dataset, filename, n_splits=100):
        self.n_splits = n_splits
        self.qsar_dataset = qsar_dataset
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

    # Similarity matrix subset:

    def get_sim_matrix_tr_samples(self, split_number, fold_number):
        internal_tr_samples = self.get_id_internal_tr_samples(split_number, fold_number)
        return self.qsar_dataset.pairwise_similarity.loc[internal_tr_samples.values, internal_tr_samples.values]

    def get_sim_matrix_ts_samples(self, split_number, fold_number):
        internal_ts_samples = self.get_id_internal_ts_samples(split_number, fold_number)
        return self.qsar_dataset.pairwise_similarity.loc[internal_ts_samples.values, internal_ts_samples.values]

    def get_sim_matrix_internal_samples(self, split_number):
        internal_samples = self.get_id_internal_samples(split_number)
        return self.qsar_dataset.pairwise_similarity.loc[internal_samples.values, internal_samples.values]

    def get_sim_matrix_external_samples(self, split_number):
        external_samples = self.get_id_external_samples(split_number)
        return self.qsar_dataset.pairwise_similarity.loc[external_samples.values, external_samples.values]


class QSARValidation:
    """ Performs QSAR validation workflow
    """

    def __init__(self, estimator, data_split, split_number, is_random_search=False):
        self.estimator = estimator
        self.is_random_search = is_random_search
        self.data_split = data_split
        self.split_number = split_number

        self.n_splits = data_split.n_splits
        self.dataset_name = data_split.qsar_dataset.name
        self.dataset_version = 'default'

    def predefined_cross_validation(self, param_grid, fit_params, folds=None, n_jobs=-1):
        """ Run cross validation in parallel with grid search or random search """
        if self.is_random_search:
            # If it is random search, creates 6 random combinations of
            #  the parameters grid/distribution for each fold
            paramGrid = ParameterSampler(param_grid, 6)
        else:
            # Regular GridSearch, obtains a combination of all possible parameters
            paramGrid = ParameterGrid(param_grid)
        print(self.estimator)

        # Find optimal threshold
        if self.estimator.algorithm_name == 'modSAR':
            internal_samples_sim = self.data_split.get_sim_matrix_internal_samples(self.split_number)
            _, threshold = GraphUtils.find_optimal_threshold(internal_samples_sim)

            fit_params['threshold'] = threshold

        """ Creats parallel tasks for the cross-validation.
        This is the same function used in the source code of GridSearchCV in sklearn.
        Parallel function will take care of all for loops defined here and will correctly
        allocate more computational resources when each for loop complete.
        Each for loop runs the function _fit_and_score defined above """
        cross_validation_results =  \
            Parallel(n_jobs=n_jobs, verbose=True, pre_dispatch='n_jobs') \
            (delayed(self._fit_and_score)(deepcopy(self.estimator), fold, params, fit_params)
             for fold in range(1, self.n_splits + 1) if folds is None or (folds is not None and fold in folds)
             for params in paramGrid)

        # After cross-validation, gather results and picks best model
        (results, cv_models) = zip(*cross_validation_results)
        results = pd.concat(results, ignore_index=True)

        bestFold = results["test_mae"].idxmin()
        # Shows parameters of the best fold
        print("Metrics for best model in cross-validation:")
        print(results.iloc[bestFold])
        best_model = cv_models[bestFold]

        # External Validation
        external_X = self.data_split.get_external_samples(self.split_number)
        external_y = self.data_split.get_external_Y(self.split_number)

        if self.estimator.algorithm_name == "modSAR":
            id_external_samples = self.data_split.get_id_external_samples(self.split_number)
            externalX_smiles = self.data_split.qsar_dataset.X_smiles.loc[id_external_samples]

            pred = best_model.predict(external_X, externalX_smiles)
        else:
            pred = best_model.predict(external_X)

        mae_external = mean_absolute_error(external_y, pred)
        rmse_external = mean_squared_error(external_y, pred) ** 0.5

        if best_model.algorithm_name in ["OplraRegularised", "OplraFeatureSelection"]:
            external_results = pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                             'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                             'fold': results.iloc[bestFold]["fold"], 'algorithm': best_model.algorithm_name,
                                             'algorithm_version': best_model.algorithm_version, 'internal': 'FALSE',
                                             'train_mae': 'NA', 'test_mae': mae_external,
                                             'train_rmse': 'NA', 'test_rmse': rmse_external, 'fit_time': 'NA',
                                             'beta': results.iloc[bestFold]['beta'],
                                             'lambda': results.iloc[bestFold]['lambda'],
                                             'no_regions': results.iloc[bestFold]['no_regions'],
                                             'no_features': results.iloc[bestFold]['no_features']},
                                            index=np.arange(1))
        elif best_model.algorithm_name in ["OplraEnsemble"]:
            external_results = pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                             'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                             'fold': results.iloc[bestFold]["fold"], 'algorithm': best_model.algorithm_name,
                                             'algorithm_version': best_model.algorithm_version, 'internal': 'FALSE',
                                             'train_mae': 'NA', 'test_mae': mae_external,
                                             'train_rmse': 'NA', 'test_rmse': rmse_external, 'fit_time': 'NA',
                                             'beta': results.iloc[bestFold]['beta'],
                                             'lambda': results.iloc[bestFold]['lambda'],
                                             'no_repeats': results.iloc[bestFold]['no_repeats'],
                                             'resampling': results.iloc[bestFold]['resampling'],
                                             'avg_no_regions': results.iloc[bestFold]['avg_no_regions'],
                                             'no_features': results.iloc[bestFold]['no_features']},
                                            index=np.arange(1))
        elif best_model.algorithm_name in ["modSAR"]:
            external_results = pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                             'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                             'fold': results.iloc[bestFold]["fold"], 'algorithm': best_model.algorithm_name,
                                             'algorithm_version': best_model.algorithm_version, 'internal': 'FALSE',
                                             'no_modules': results.iloc[bestFold]['no_modules'],
                                             'no_classes': results.iloc[bestFold]['no_classes'],
                                             'threshold': results.iloc[bestFold]['threshold'],
                                             'train_mae': 'NA', 'test_mae': mae_external,
                                             'train_rmse': 'NA', 'test_rmse': rmse_external, 'fit_time': 'NA',
                                             'beta': results.iloc[bestFold]['beta'],
                                             'lambda': results.iloc[bestFold]['lambda']},
                                            index=np.arange(1))
        else:
            external_results = pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                             'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                             'fold': results.iloc[bestFold]["fold"], 'algorithm': best_model.algorithm_name,
                                             'algorithm_version': best_model.algorithm_version, 'internal': 'FALSE',
                                             'no_modules': None,
                                             'no_classes': None,
                                             'threshold': None,
                                             'train_mae': 'NA', 'test_mae': mae_external,
                                             'train_rmse': 'NA', 'test_rmse': rmse_external, 'fit_time': 'NA',
                                             'beta': None,
                                             'lambda': None},
                                            index=np.arange(1))

        results = pd.concat([results, external_results], ignore_index=True)

        return results, best_model

    def _fit_and_score(self, estimator, fold, params, fit_params):
        """A iteration of cross-validation with algorithm <estimator>, fold number <fold> and samples in
        training/testing defined in <cvIter>, run with parameters in <params>

        :param estimator:
        :param fold:
        :param params:
        :return:
        """
        print("Iteration: %d/%d" % (fold, self.n_splits))
        # Sets parameters for the algorithms, as defined by the grid search
        estimator.set_params(**params)

        # Runs the algorithm in the predefined split of this Cross-Validation iteration
        trainX = self.data_split.get_train_samples_in_fold(self.split_number, fold)
        trainY = self.data_split.get_train_Y_in_fold(self.split_number, fold)

        testX = self.data_split.get_test_samples_in_fold(self.split_number, fold)
        testY = self.data_split.get_test_Y_in_fold(self.split_number, fold)

        start = time.time()
        if estimator.algorithm_name == "modSAR":
            estimator.solver_def = get_solver_definition(estimator.solver_name)  # CPLEX or GLPK

            # Obtain smiles codes for samples in training
            internal_tr_samples = self.data_split.get_id_internal_tr_samples(self.split_number, fold)
            trainX_smiles = self.data_split.qsar_dataset.X_smiles.loc[internal_tr_samples]
            sim_matrix = self.data_split.qsar_dataset.pairwise_similarity.loc[internal_tr_samples, internal_tr_samples]

            print(self.data_split.qsar_dataset)
            print(trainX.shape)

            estimator.fit(trainX, trainY, sim_matrix, trainX_smiles,
                          threshold=fit_params['threshold'],
                          k=fit_params['k'])
        elif estimator.algorithm_name == "OplraRegularised":
            trainY = trainY['pchembl_value']  # Get series
            estimator.solver_def = get_solver_definition(estimator.solver_name)
            if self.estimator.algorithm_version == "v1_1":
                estimator.fit(trainX, trainY, f_star=fit_params['fStar'])
            else:
                estimator.fit(trainX, trainY)
        else:
            estimator.fit(trainX, trainY)

        end = time.time()

        if estimator.algorithm_name == "modSAR":
            train_predicted = estimator.predict(trainX, trainX_smiles)
        else:
            train_predicted = estimator.predict(trainX)

        trainMAE = mean_absolute_error(trainY, train_predicted)
        trainRMSE = mean_squared_error(trainY, train_predicted) ** 0.5

        if estimator.algorithm_name == "modSAR":
            internal_ts_samples = self.data_split.get_id_internal_ts_samples(self.split_number, fold)
            testX_smiles = self.data_split.qsar_dataset.X_smiles.loc[internal_ts_samples]
            test_predicted = estimator.predict(testX, testX_smiles)
        else:
            test_predicted = estimator.predict(testX)

        testMAE = mean_absolute_error(testY, test_predicted)
        testRMSE = mean_squared_error(testY, test_predicted) ** 0.5

        result = self.get_results_df(estimator, fold, start, end, trainMAE, testMAE, trainRMSE, testRMSE, params)

        return result, estimator

    def get_results_df(self, estimator, fold, start, end, trainMAE, testMAE, trainRMSE, testRMSE, params):
        if estimator.algorithm_name in ["OplraRegularised", "OplraFeatureSelection"]:
            return pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                 'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                 'fold': 'fold%d' % (fold + 1), 'algorithm': self.estimator.algorithm_name,
                                 'algorithm_version': self.estimator.algorithm_version, 'internal': 'TRUE',
                                 'train_mae': trainMAE, 'test_mae': testMAE,
                                 'train_rmse': trainRMSE, 'test_rmse': testRMSE,
                                 'fit_time': (end - start), 'beta': params['beta'], 'lambda': params['lam'],
                                 'no_regions': estimator.final_model.number_regions,
                                 'no_features': len(estimator.final_model.get_selected_features())},
                                index=np.arange(1))
        elif estimator.algorithm_name in ["OplraEnsemble"]:
            return pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                 'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                 'fold': 'fold%d' % (fold + 1), 'algorithm': self.estimator.algorithm_name,
                                 'algorithm_version': self.estimator.algorithm_version, 'internal': 'TRUE',
                                 'train_mae': trainMAE, 'test_mae': testMAE,
                                 'train_rmse': trainRMSE, 'test_rmse': testRMSE,
                                 'fit_time': (end - start), 'beta': params['beta'], 'lambda': params['lam'],
                                 'no_repeats': params['noRepeats'], 'resampling': params['resampling'],
                                 'avg_no_regions': estimator.avg_number_regions(),
                                 'no_features': len(estimator.get_selected_features())},
                                index=np.arange(1))
        elif estimator.algorithm_name in ["modSAR"]:
            return pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                 'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                 'fold': fold, 'algorithm': self.estimator.algorithm_name,
                                 'algorithm_version': self.estimator.algorithm_version, 'internal': 'TRUE',
                                 'no_modules': estimator.number_modules, 'no_classes': estimator.number_classes,
                                 'threshold': estimator.threshold,
                                 'train_mae': trainMAE, 'test_mae': testMAE,
                                 'train_rmse': trainRMSE, 'test_rmse': testRMSE,
                                 'fit_time': (end - start), 'beta': params['beta'], 'lambda': params['lam']},
                                index=np.arange(1))
        else:
            return pd.DataFrame({'splitStrategy': 1, 'splitNumber': self.split_number,
                                 'dataset': self.dataset_name, 'datasetVersion': self.dataset_version,
                                 'fold': 'fold%d' % (fold + 1), 'algorithm': self.estimator.algorithm_name,
                                 'algorithmVersion': self.estimator.algorithm_version, 'internal': 'TRUE',
                                 'train_mae': trainMAE, 'test_mae': testMAE, 'train_rmse': trainRMSE, 'test_rmse': testRMSE,
                                 'fit_time': (end - start), 'params': str(params)},
                                index=np.arange(1))
