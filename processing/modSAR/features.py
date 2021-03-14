"""Feature filter selection"""
import rdkit

import numpy as np
import pandas as pd

from .utils import print_progress_bar


def convert_to_morgan_fingerprints(X, nBits=1024):
    """ Convert ECFP4 features matrix (a DataFrame with `nBits` columns)
        to morgan fingerprints with RDKit library
    """

    if type(X) == pd.DataFrame:
        X_array = X.values
    elif type(X) == np.ndarray:
        X_array = X
    else:
        raise ValueError("Unknown type (X) = %s" % type(X))

    X_fingerprints = list()
    for row in X_array:
        bitvect = rdkit.DataStructs.cDataStructs.ExplicitBitVect(nBits, False)
        for i, col in enumerate(row):
            bitvect.SetBit(i)

        X_fingerprints.append(bitvect)

    return X_fingerprints


def apply_feature_filter(X):
    """Remove features with near zero variance and features with high correlation"""

    cols_to_remove = near_zero_variance(X)
    X = X.loc[:, ~cols_to_remove].copy()

    cols_to_remove = find_correlation_exact(X)
    X = X.loc[:, ~cols_to_remove].copy()
    return X


def near_zero_variance(df, freq_cut=95 / 5, unique_cut=10):
    """Closely follows caret::nearZeroVar implementation

    https://rdrr.io/cran/caret/src/R/nearZeroVar.R
    """

    def nzv(series, freq_cut=95 / 5, unique_cut=10):
        # Round to 5 decimal places to emulate default behaviour in R?
        unique_values = series.round(5).value_counts()

        if len(unique_values) == 1:
            # Is Zero Variance
            return True

        freq_ratio = unique_values.iloc[0] / unique_values.iloc[1]
        percent_unique = len(unique_values) / series.shape[0]

        return (freq_ratio > freq_cut) and (percent_unique <= unique_cut)

    return df.apply(lambda x: nzv(x, freq_cut, unique_cut))


def find_correlation_exact(df, cutoff=0.9, verbose=False):
    """Closely follows caret::findCorrelation_fast implementation

    https://rdrr.io/cran/caret/src/R/findCorrelation.R
    """
    mat = df.corr()

    # Reorder correlation matrix per average correlation mean
    correlation_order = mat.abs().mean().sort_values(ascending=False).index

    # Auxiliar correlation matrix: mat2. Diagonals are set to None
    mat2 = mat.copy()
    for i in correlation_order:
        mat2.loc[i, i] = None

    n_cols = len(correlation_order)
    total_comparisons = n_cols * (n_cols - 1)

    cols_to_remove = pd.Series([False] * n_cols, index=correlation_order)
    count = 0
    for i, i_col in enumerate(correlation_order):
        print_progress_bar(count, total_comparisons)
        if mat2.isnull().all().all():
            print_progress_bar(total_comparisons, total_comparisons)
            break
        if cols_to_remove[i_col]:
            count += n_cols
            continue
        for j, j_col in enumerate(correlation_order):
            count += 1
            if j <= i:
                continue
            print_progress_bar(count, total_comparisons)
            if not (cols_to_remove[i_col] or cols_to_remove[j_col]):
                if mat.loc[i_col, j_col] > cutoff:
                    # Remove column with largest mean absolute correlation
                    mn1 = mat2.loc[i_col].mean()
                    mn2 = mat2.loc[j_col].mean()
                    if verbose:
                        print("-- MN1 (i=%s) : %.3f" % (i_col, mn1))
                        print("-- MN2 (j=%s) : %.3f" % (j_col, mn2))

                    if mn1 > mn2:
                        if verbose:
                            print("-- Removing %s" % (i_col))
                        cols_to_remove.loc[i_col] = True
                        mat2.at[i_col, :] = None
                        mat2[[i_col]] = None
                    else:
                        if verbose:
                            print("-- Removing %s" % (j_col))
                        cols_to_remove.loc[j_col] = True
                        mat2.at[j_col, :] = None
                        mat2[[j_col]] = None

    return cols_to_remove
