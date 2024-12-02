import pandas as pd
import numpy as np

from ..dataset import Dataset


class CIC2017Dataset(Dataset):

    def get_dataset_name(self) -> str:
        return "CIC_2017"

    def get_raw_files_directory_name(self) -> str:
        return "CIC_2017"

    def clear_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # data = self.best_features_func(data)
        data = self.drop_one_features(data)
        data = self.drop_duplicate_columns(data)

        data = self.drop_bad_elements(data)
        x = data.drop(columns=[" Label"], axis=1)
        y = data[" Label"]

        labels = set(y)

        labels.remove("BENIGN")

        print(f"labels: {labels}")

        y = self.replace(y=y, list_B_columns=["BENIGN"], list_M_columns=labels)

        x = self.drop_bad_elements_x(x)

        return x, y

    def drop_bad_elements(self, data: pd.DataFrame) -> pd.DataFrame:
        """drop_bad_elements

        This method is used to drop bad elements.

        Output:
            None
        """
        pass
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0)

        # data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

        return data

    def drop_bad_elements_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """drop_bad_elements_x

        This method is used to drop bad elements.

        Output:
            None
        """
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        x[np.isneginf(x)] = 0
        # x[x < 0] = 0

        return x

    def drop_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """drop_duplicate_columns

        This method is used to drop duplicate columns.

        Output:
            None
        """
        # drop duplicate rows
        data.drop_duplicates(keep="first")
        data.reset_index(drop=True, inplace=True)

        return data

    def drop_one_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """drop_one_features

        This method is used to drop one features.

        Output:
            None
        """
        # drop one variable features
        one_variable_list = []
        for i in data.columns:
            if data[i].value_counts().nunique() < 2:
                one_variable_list.append(i)
        data.drop(one_variable_list, axis=1, inplace=True)

        return data

    def replace(self, y: pd.DataFrame, list_B_columns: list = None, list_M_columns: list = None) -> pd.DataFrame:
        """replace

        This method is responsible for replacing the label's values.
        Standardize those values to Benign for normal traffic and Malicious for malicious traffic.

        Parameters:
            list_B_columns (list): List of columns to replace the Benign label.
            list_M_columns (list): List of columns to replace the Malicious label.
        Output:
            None
        """
        with pd.option_context("future.no_silent_downcasting", True):
            if not type(list_B_columns) is None:
                y.replace(to_replace=list_B_columns, value=0, inplace=True)
            if not type(list_M_columns) is None:
                y.replace(to_replace=list_M_columns, value=1, inplace=True)
        y = y.astype("int8")

        return y


if __name__ == '__main__':
    dataset = CIC2017Dataset(load_preprocessed_data=False, seed=42)
