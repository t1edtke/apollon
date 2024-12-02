import glob
import logging
import os
from abc import ABC, abstractmethod
from typing import List
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.model_selection import train_test_split


class Dataset(ABC):

    raw_data_directory = "live/data_raw"
    preprocessed_data_directory = "live/data_preprocessed"

    dtypes = None

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __init__(self, load_preprocessed_data: bool, seed: int, files: List[str] = None) -> None:
        start_time = time.time()
        if load_preprocessed_data and not files:
            X, y = self.load_preprocessed_data()
        else:
            raw_data = self.load_raw_data(seed=seed, files_filter=files)
            X, y = self.clear_data(raw_data)
            if not files:
                self.save_preprocessed_data(X, y)
        self.transform(X, y, seed)
        logging.debug(f"Dataset {self.get_dataset_name()} loaded in {time.time() - start_time:.2f} seconds")

    def load_preprocessed_data(self) -> (pd.DataFrame, pd.DataFrame):
        logging.debug(f"Loading preprocessed data for {self.get_dataset_name()}")
        directory = f"{self.preprocessed_data_directory}/{self.get_dataset_name()}"
        X = pd.read_csv(f"{directory}/data.csv")
        y = pd.read_csv(f"{directory}/labels.csv").values.ravel()
        return X, y

    def load_raw_data(self, seed: int, files_filter: List[str]) -> pd.DataFrame:
        logging.debug(f"Loading raw data for {self.get_dataset_name()}")
        directory = f"{self.raw_data_directory}/{self.get_raw_files_directory_name()}"
        files = sorted(glob.glob(f"{directory}/**/*.csv", recursive=True))
        files = [file for file in files if files_filter is None or any(f == file.removeprefix(f"{directory}/") for f in files_filter)]
        data = pd.concat([pd.read_csv(file, dtype=self.dtypes) for file in files])
        return data.sample(frac=1, random_state=seed)

    @abstractmethod
    def clear_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame, y: pd.DataFrame, seed: int) -> None:
        logging.debug(f"Transforming data for {self.get_dataset_name()}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

        robust_scaler = RobustScaler()
        X_train = robust_scaler.fit_transform(X_train)
        X_test = robust_scaler.transform(X_test)

        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)] = 0

        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_preprocessed_data(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        logging.debug(f"Saving preprocessed data for {self.get_dataset_name()}")
        directory = f"{self.preprocessed_data_directory}/{self.get_dataset_name()}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        X.to_csv(f"{directory}/data.csv", index=False)
        y.to_csv(f"{directory}/labels.csv", index=False)

    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    @abstractmethod
    def get_raw_files_directory_name(self) -> str:
        pass
