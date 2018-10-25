class Preprocessing:

    def preprocess_activities(self, bioactivities_df):
        """Pre-process ChEMBL bioactivities DataFrame"""

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
