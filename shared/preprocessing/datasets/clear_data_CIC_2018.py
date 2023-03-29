# ========================== Clear Data - CIC IDS 2018 ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ===============================================================================

# ==================> Imports
import pandas as pd
import shared.preprocessing.clear_data as cd

from sklearn.feature_selection import SelectKBest, chi2


# ==================> Classes
class ClearDataCIC2018(cd.ClearData):
    def __init__(self, df: pd.DataFrame, do_save: bool, seed: int) -> None:
        """__init__

        This method is used to initialize the ClearDataCIC2018 class.

        Parameters:
            df: pd.DataFrame
            do_save: bool
        Output:
            None
        """
        super().__init__(df=df, seed=seed)
        self.do_save = do_save

    # Override
    def clear_data(self) -> None:
        """clear_data

        This method is used to clear the data of the CIC 2018 dataset.

        Output:
            None
        """
        self.df.drop(['Timestamp'], axis=1, inplace=True)
        self.reduce_tam()
        # self.best_features_func()
        self.drop_one_features()
        self.drop_duplicate_columns()

        self.x = self.df.drop(["Label"], axis=1)
        self.y = self.df["Label"]

        labels = set(self.y)

        labels.remove("BENIGN")

        print(f"labels: {labels}")

        self.replace(list_B_columns=["Benign"], list_M_columns=labels)
        self.drop_bad_elements_x()

        if self.do_save:
            self.save_data()

    # Override
    def save_data(self):
        aux_df = self.df
        aux_df.drop(["Label"], axis=1, inplace=True)

        aux_y = pd.DataFrame(self.y, columns=['Label'])
        aux_df = pd.concat([aux_df, aux_y], axis=1)

        aux_df.to_csv('./shared/data_prep/CIC18/CIC18.csv', index=False)
        aux_y.to_csv('./shared/data_prep/CIC18/CIC18_y.csv', index=False)
