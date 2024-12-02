from abc import ABC, abstractmethod

import numpy as np
from sklearn import metrics
from sklearn.utils._param_validation import HasMethods

import datasets
from intrusion_detection_systems import models


class Metrics(ABC):

    def __init__(self, model: HasMethods("predict"), X_train, y_train, X_test, y_test) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        if self.X_train is not None:
            self.train_predictions = model.predict(self.X_train)
        self.test_predictions = model.predict(self.X_test)

    @abstractmethod
    def report(self) -> None:
        pass

    def train_accuracy(self) -> float:
        return metrics.accuracy_score(self.y_train, self.train_predictions)

    def test_accuracy(self) -> float:
        return metrics.accuracy_score(self.y_test, self.test_predictions)

    def train_precision(self) -> float:
        return metrics.precision_score(self.y_train, self.train_predictions)

    def test_precision(self) -> float:
        return metrics.precision_score(self.y_test, self.test_predictions)

    def train_recall(self) -> float:
        return metrics.recall_score(self.y_train, self.train_predictions)

    def test_recall(self) -> float:
        return metrics.recall_score(self.y_test, self.test_predictions)

    def train_f1(self) -> float:
        return metrics.f1_score(self.y_train, self.train_predictions)

    def test_f1(self) -> float:
        return metrics.f1_score(self.y_test, self.test_predictions)

    def train_auc(self) -> float:
        return metrics.roc_auc_score(self.y_train, self.train_predictions)

    def test_auc(self) -> float:
        return metrics.roc_auc_score(self.y_test, self.test_predictions)

    def train_confusion_matrix(self) -> float:
        return metrics.confusion_matrix(self.y_train, self.train_predictions)

    def test_confusion_matrix(self) -> float:
        return metrics.confusion_matrix(self.y_test, self.test_predictions)

    def train_classification_report(self) -> str | dict:
        return metrics.classification_report(self.y_train, self.train_predictions)

    def test_classification_report(self) -> str | dict:
        return metrics.classification_report(self.y_test, self.test_predictions)
