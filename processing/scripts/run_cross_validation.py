"""Run predefined cross validation for one data split"""

import os
import argparse

from modSAR.dataset import QSARDatasetIO
from modSAR.network_algorithms import ModSAR

from scripts.validation import DataSplit, QSARValidation

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to run cross-validation')

    parser.add_argument('--dataset', default=None, nargs='?', const=1,
                        type=str, help='Dataset name. Ex.: CHRM3, hDHFR.')
    parser.add_argument('--datasplit', default=None, nargs='?', const=1,
                        type=int, help='Predefined datasplit number, one of: [1, 2, 3, 4, 5]')
    parser.add_argument('--datafolder', default='/mnt/data', nargs='?', const=1,
                        type=str, help='Folder where data sets and data splits are stored.')
    parser.add_argument('--algorithm', default='modSAR', nargs='?', const=1,
                        type=str, help='One of the algorithms: ["modSAR"]')
    parser.add_argument('--savefolder', default='/mnt/data/results', nargs='?', const=1,
                        type=str, help='Folder where results should be saved.')
    arguments = vars(parser.parse_args())

    return arguments


def run_cv(algorithm_name, dataset_name, split_number, data_folder):
    filepath = os.path.join(data_folder, '%s.xlsx' % dataset_name)

    if algorithm_name.lower() == "modsar":
        qsar_dataset = QSARDatasetIO.load(filepath, dataset_name)
    else:
        qsar_dataset = QSARDatasetIO.load(filepath, dataset_name, calculate_similarity=False)

    splits_filepath = os.path.join(data_folder, '%s_splits.xlsx' % dataset_name)
    data_split = DataSplit(qsar_dataset, filename=splits_filepath)

    if algorithm_name.lower() == 'modsar':
        algorithm = ModSAR()
        param_grid = {'beta': [0.03], 'lam': [0.005, 0.05, 0.1], 'solver_name': ['cplex']}
        fit_params = {'k': 0}
    elif algorithm_name.lower() == "rf" or algorithm_name.lower() == "randomforest":
        algorithm = RandomForestRegressor()
        algorithm.algorithm_name = 'RandomForest'
        algorithm.algorithm_version = ""
        param_grid = {}
        fit_params = {}

    qsar_validation = QSARValidation(estimator=algorithm,
                                     data_split=data_split,
                                     split_number=split_number)
    results, best_model = qsar_validation.predefined_cross_validation(param_grid, fit_params=fit_params, n_jobs=1)

    return results, best_model


if __name__ == '__main__':

    args = parse_arguments()
    print('[PARAMETERS] \n')
    print(' algorithm == %s \n dataset == %s \n datasplit == %s \n datafolder == %s \n savefolder == %s' %
          (args['algorithm'], args['dataset'], args['datasplit'], args['datafolder'], args['savefolder']))

    results, best_model = run_cv(args['algorithm'],
                                 args['dataset'],
                                 args['datasplit'],
                                 args['datafolder'])

    # Subfolders
    filepath = os.path.join(args['savefolder'], args['dataset'])
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = 'results_%s_split%02d_alg_%s.xlsx' % (args['dataset'], args['datasplit'], args['algorithm'].lower())
    results.to_excel(os.path.join(filepath, filename))

    filename = '%s_split%02d_alg_%s.joblib' % (args['dataset'], args['datasplit'], args['algorithm'].lower())
    joblib.dump(best_model, filename)
