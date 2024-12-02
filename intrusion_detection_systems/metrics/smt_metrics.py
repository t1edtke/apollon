import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.utils._param_validation import HasMethods

from intrusion_detection_systems import models


class SMTMetrics:

    def __init__(self, model: HasMethods("predict")) -> None:
        self.model = model

    def report(self) -> None:

        n = 5

        cv = ShuffleSplit(n_splits=n, test_size=0.2,
                          random_state=self.model.seed)

        model_predictions = self.model.predict(self.model.dataset.X_test)
        confusion_matrix = metrics.confusion_matrix(
            self.model.dataset.y_test, model_predictions)
        classification = metrics.classification_report(
            self.model.dataset.y_test, model_predictions)

        scoring = ('accuracy', 'roc_auc', 'f1_macro', 'recall_macro')

        scores = cross_validate(self.model.model,
                                self.model.dataset.X_train, self.model.dataset.y_train, cv=cv, scoring=scoring, return_train_score=True)

        cv_accuracy_te = np.mean(scores['test_accuracy'])
        std_accuracy_te = np.std(scores['test_accuracy'])

        cv_roc_te = np.mean(scores['test_roc_auc'])
        std_roc_te = np.std(scores['test_roc_auc'])

        cv_f1_te = np.mean(scores['test_f1_macro'])
        std_f1_te = np.std(scores['test_f1_macro'])

        cv_dr_te = np.mean(scores['test_recall_macro'])
        std_dr_te = np.std(scores['test_recall_macro'])

        cv_accuracy_tr = np.mean(scores['train_accuracy'])
        std_accuracy_tr = np.std(scores['train_accuracy'])

        cv_roc_tr = np.mean(scores['train_roc_auc'])
        std_roc_tr = np.std(scores['train_roc_auc'])

        cv_f1_tr = np.mean(scores['train_f1_macro'])
        std_f1_tr = np.std(scores['train_f1_macro'])

        cv_dr_tr = np.mean(scores['train_recall_macro'])
        std_dr_tr = np.std(scores['train_recall_macro'])

        print(
            f'\n============================== {self.model.dataset.get_dataset_name()} Model Evaluation {self.model.get_model_name()} ==============================\n')
        print(
            f"[TEST]\tCross Validation Mean and std Score for F1:\t{cv_f1_te}\t{std_f1_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for accuracy:\t{cv_accuracy_te}\t{std_accuracy_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for roc_auc:\t{cv_roc_te}\t{std_roc_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for detection rate:\t{cv_dr_te}\t{std_dr_te}\n")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for F1:\t{cv_f1_tr}\t{std_f1_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for accuracy:\t{cv_accuracy_tr}\t{std_accuracy_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for roc_auc:\t{cv_roc_tr}\t{std_roc_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for detection rate:\t{cv_dr_tr}\t{std_dr_tr}\n")
        print(f"Confusion matrix: \n{confusion_matrix}")
        print(f"Classification report: \n{classification}")
