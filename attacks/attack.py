from abc import ABC, abstractmethod

import numpy as np
from art.estimators.classification import SklearnClassifier
from sklearn.utils._param_validation import HasMethods

import datasets


class Attack(ABC):

    def generate_adversarial_examples(self, model: HasMethods("predict"), dataset: datasets.Dataset, n: int = 500) -> (np.ndarray, np.ndarray):
        try:
            classifier = SklearnClassifier(model=model)
        except:
            classifier = model
        attack = self.get_attack(classifier)

        X_test_malicious = dataset.X_test[dataset.y_test == 1][:n]

        X_adv, y_adv = attack.generate(
            x=X_test_malicious,
            y=np.zeros(X_test_malicious.shape[0]),
        ), np.ones(X_test_malicious.shape[0])

        return X_adv, y_adv

    @abstractmethod
    def get_attack(self, classifier: SklearnClassifier):
        pass
