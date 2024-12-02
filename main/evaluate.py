import logging
from typing import Type

import numpy as np
from sklearn.utils._param_validation import HasMethods

import attacks
import datasets
from intrusion_detection_systems import apollon, metrics


def generate_train_data(dataset: datasets.Dataset, attack_type: Type[attacks.Attack], model: HasMethods("predict"), n: int = 500):
    X_train_benign = dataset.X_train[dataset.y_train == 0][:n]
    y_train_benign = dataset.y_train[dataset.y_train == 0][:n]

    X_train_malicious = dataset.X_train[dataset.y_train == 1][:n]
    y_train_malicious = dataset.y_train[dataset.y_train == 1][:n]

    X_train_adv, y_train_adv = attack_type().generate_adversarial_examples(model=model, dataset=dataset, n=n)

    X_train_with_adv = np.concatenate([X_train_benign[:n], X_train_adv])
    y_train_with_adv = np.concatenate([y_train_benign[:n], y_train_adv])

    X_train_without_adv = np.concatenate([X_train_benign[:n], X_train_malicious[:n]])
    y_train_without_adv = np.concatenate([y_train_benign[:n], y_train_malicious[:n]])

    return (X_train_without_adv, y_train_without_adv), (X_train_with_adv, y_train_with_adv)


def generate_test_data(dataset: datasets.Dataset, attack_type: Type[attacks.Attack], model: HasMethods("predict"), n: int = 500):
    X_test_benign = dataset.X_test[dataset.y_test == 0][:n]
    y_test_benign = dataset.y_test[dataset.y_test == 0][:n]

    X_test_malicious = dataset.X_test[dataset.y_test == 1][:n]
    y_test_malicious = dataset.y_test[dataset.y_test == 1][:n]

    X_test_adv, y_test_adv = attack_type().generate_adversarial_examples(model=model, dataset=dataset, n=n)

    X_test_with_adv = np.concatenate([X_test_benign[:n], X_test_adv])
    y_test_with_adv = np.concatenate([y_test_benign[:n], y_test_adv])

    X_test_without_adv = np.concatenate([X_test_benign[:n], X_test_malicious[:n]])
    y_test_without_adv = np.concatenate([y_test_benign[:n], y_test_malicious[:n]])

    return (X_test_without_adv, y_test_without_adv), (X_test_with_adv, y_test_with_adv)

def evaluate_model_on_testdata_using_metric(model: HasMethods("predict"), test_data, metric_type: Type[metrics.Metrics]):
    metric_non_adv = metric_type(model=model, X_train=None, y_train=None, X_test=test_data[0][0], y_test=test_data[0][1])
    metric_non_adv.report()
    metric_adv = metric_type(model=model, X_train=None, y_train=None, X_test=test_data[1][0], y_test=test_data[1][1])
    metric_adv.report()


def evaluate_apollon_on_testdata_using_metric(apollon: apollon.MABPredictor, testdatas, metric_type: Type[metrics.Metrics]):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for i in range(len(testdatas)):

        metric_adv = metric_type(model=apollon, X_train=None, y_train=None, X_test=testdatas[i][1][0], y_test=testdatas[i][1][1])

        accuracies.append(metric_adv.test_accuracy())
        precisions.append(metric_adv.test_precision())
        recalls.append(metric_adv.test_recall())
        f1s.append(metric_adv.test_f1())
        aucs.append(metric_adv.test_auc())

    logging.info(f"Average accuracy: {np.mean(accuracies)}")
    logging.info(f"Average precision: {np.mean(precisions)}")
    logging.info(f"Average recall: {np.mean(recalls)}")
    logging.info(f"Average f1: {np.mean(f1s)}")
    logging.info(f"Average auc: {np.mean(aucs)}")


def evaluate_apollon_on_non_adv_testdata_using_metric(apollon: apollon.MABPredictor, testdatas, metric_type: Type[metrics.Metrics]):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for i in range(len(testdatas)):

        metric_non_adv = metric_type(model=apollon, X_train=None, y_train=None, X_test=testdatas[i][0][0], y_test=testdatas[i][0][1])

        accuracies.append(metric_non_adv.test_accuracy())
        precisions.append(metric_non_adv.test_precision())
        recalls.append(metric_non_adv.test_recall())
        f1s.append(metric_non_adv.test_f1())
        aucs.append(metric_non_adv.test_auc())

    logging.info(f"Average accuracy: {np.mean(accuracies)}")
    logging.info(f"Average precision: {np.mean(precisions)}")
    logging.info(f"Average recall: {np.mean(recalls)}")
    logging.info(f"Average f1: {np.mean(f1s)}")
    logging.info(f"Average auc: {np.mean(aucs)}")
