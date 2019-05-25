import numpy as np
import pandas as pd


class Preprocessing:
    """Preprocess data obtained via a DataSource (.csv, .xls file or ChEMBL)

    The bioactivity data is filtered, duplicated and invalid entries are handled and
      molecular descriptors are calculated.
    """

    def __init__(self, compound_id_column, activity_column,
                 apply_chembl_filter=True, remove_duplicated=True):
        missing_attributes = []
        if (activity_column is None):
            missing_attributes.append(['activity_column'])
        if (compound_id_column is None):
            missing_attributes.append(['compound_id_column'])

        if (len(missing_attributes) > 1):
            raise ValueError('Required attribute(s) missing: %s' % (', '.join(missing_attributes)))

        self.compound_id_column = compound_id_column
        self.activity_column = activity_column
        self.remove_duplicated = remove_duplicated
        self.apply_chembl_filter = apply_chembl_filter

    def do(self, df):
        if self.apply_chembl_filter:
            df = apply_chembl_filter(df)
        if self.remove_duplicated:
            df = remove_duplicated(df, compound_id_column=self.compound_id_column,
                                   activity_column=self.activity_column)
        return df


def apply_chembl_filter(bioactivities_df):
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


def remove_duplicated(bioactivities_df, compound_id_column, activity_column):
    """
    Removes invalid compounds and handles duplicated entries
    """

    # Mark duplicated entries for removal
    def mark_to_remove(data, activity_column):
        """Mark data for removal if standard deviation of duplicated entries is above 1"""

        activity_value = data[activity_column].astype(float)
        std_deviation = np.std(activity_value)
        number_compounds = data.shape[0]

        # It is duplicated
        result_df = pd.Series(
            {'number_compounds': number_compounds,
             'standard_deviation': std_deviation,
             activity_column: np.median(activity_value),
             'duplicated': number_compounds > 1,
             'mark_to_remove': std_deviation > 1})
        return result_df

    grouped_dataset = bioactivities_df.groupby(compound_id_column)
    grouped_dataset = grouped_dataset.apply(lambda x: mark_to_remove(x, activity_column))

    # The resulting dataset, clean_df, does not contain duplicated entries
    merged_df = pd.merge(bioactivities_df, grouped_dataset.reset_index(),
                         on='parent_molecule_chembl_id')
    merged_df = merged_df[~merged_df['mark_to_remove']]
    clean_df = merged_df.groupby(compound_id_column).head(1)

    # Set ID as DataFrame index
    clean_df.set_index(compound_id_column, inplace=True)
    return clean_df
