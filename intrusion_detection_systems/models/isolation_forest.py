from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


def CustomizedIsolationForest(random_state: Any) -> IsolationForest:
    iso = IsolationForest(
        # n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    customize(iso)
    return iso

def customize(iso: IsolationForest):
    # Map anomaly scores -1 to 1 and 1 to 0
    # Train on only benign data
    fit = iso.fit
    iso.fit = lambda X, y: fit(X[y == 0], y[y == 0])
    predict = iso.predict
    iso.predict = lambda X: np.where(predict(X) == -1, 1, 0)
    fit_predict = iso.fit_predict
    iso.fit_predict = lambda X, y: np.where(fit_predict(X[y == 0], y[y == 0]) == -1, 1, 0)

    # n_classes_ is required for adversarial-robustness-toolbox
    iso.n_classes_ = 2



if __name__ == '__main__':
    import attacks
    import datasets
    from intrusion_detection_systems import metrics, models
    import main.evaluate_models.evaluate as evaluate

    seed = 42
    n = 10
    cic_2017_subsample = datasets.CIC2017SubSample(load_preprocessed_data=False, seed=seed)

    ifo = models.IF(random_state=seed)
    ifo.fit(cic_2017_subsample.X_train, cic_2017_subsample.y_train)
    ifo_testdata = evaluate.generate_test_data(dataset=cic_2017_subsample, attack_type=attacks.HSJA, model=ifo, n=n)
    evaluate.evaluate_model_on_testdata_using_metric(model=ifo, test_data=ifo_testdata, metric_type=metrics.BasicMetrics)
    y_pred = ifo.predict(cic_2017_subsample.X_test)
    print(np.unique(y_pred, return_counts=True))
