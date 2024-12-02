import random
from abc import ABC, abstractmethod
import logging
import time
from typing import List, Any

import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils._param_validation import HasMethods

import datasets
from intrusion_detection_systems import models, metrics
from intrusion_detection_systems.models import isolation_forest


class MABTrainer:

    kmeans: KMeans = None

    def __init__(self, arms: List[HasMethods(["fit", "predict"])], n_clusters: int, random_state: Any, scorer) -> None:
        super().__init__()
        self.arms = [[clone(arm, safe=False) for arm in arms] for _ in range(n_clusters)]  # Fixed: each cluster has its own set of arms
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scorer = scorer
        self.rewards = np.zeros((n_clusters, len(arms)))

        # Customize Isolation Forests since previous customization got lost during cloning
        for cluster_n in range(n_clusters):
            for arm in self.arms[cluster_n]:
                if isinstance(arm, IsolationForest):
                    isolation_forest.customize(arm)

    def fit(self, X, y) -> None:
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_assignments = self.kmeans.fit_predict(X)

        for cluster_n in range(self.n_clusters):
            cluster_mask = cluster_assignments == cluster_n
            cluster_X_train = X[cluster_mask]
            cluster_y_train = y[cluster_mask]
            logging.info(f"Cluster {cluster_n}: {cluster_y_train.shape[0]} samples: {np.unique(cluster_y_train, return_counts=True)}")

            for arm_n, arm in enumerate(self.arms[cluster_n]):
                logging.info(f"Training {arm.__class__.__name__} on cluster {cluster_n}")
                start_fit = time.time()
                arm.fit(cluster_X_train, cluster_y_train)  # Fixed: train each arm on the entire cluster instead of only the samples of the corresponding class

                logging.info(f"Setting reward for {arm.__class__.__name__} on cluster {cluster_n}")
                start_cv = time.time()
                self.rewards[cluster_n, arm_n] = cross_val_score(arm, cluster_X_train, cluster_y_train, scoring=self.scorer, cv=2).mean()

                logging.info(f"Time to fit: {start_cv - start_fit} s")
                logging.info(f"Time to cross validate: {time.time() - start_cv} s")
                logging.info(f"Setting reward for {arm.__class__.__name__} on cluster {cluster_n}: {self.rewards[cluster_n, arm_n]}")


class MABPredictor(ABC):

    def __init__(self, mab: MABTrainer):
        self.mab = mab

    def predict(self, X, return_arms: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        cluster_assignments = self.mab.kmeans.predict(X)

        arm_assignments = np.zeros(X.shape[0], dtype=int)
        for x_n, cluster_n in enumerate(cluster_assignments):
            arm_assignments[x_n] = self.select_arm(cluster_n)

        y_pred = np.zeros(X.shape[0], dtype=int)
        for cluster_n in range(self.mab.n_clusters):
            cluster_mask = cluster_assignments == cluster_n
            for arm_n, arm in enumerate(self.mab.arms[cluster_n]):
                arm_mask = arm_assignments == arm_n
                arm_X_test = X[cluster_mask & arm_mask]
                if len(arm_X_test) > 0:
                    y_pred[cluster_mask & arm_mask] = arm.predict(arm_X_test)

        if return_arms:
            arm_assignments_by_cluster = [
                [
                    np.sum(
                        (cluster_assignments == cluster_n) & (arm_assignments == arm_n)
                    )
                    for arm_n in range(len(self.mab.arms[cluster_n]))]
                for cluster_n in range(self.mab.n_clusters)
            ]
            return y_pred, np.array(arm_assignments_by_cluster)
        else:
            return y_pred

    @abstractmethod
    def select_arm(self, cluster_n: int) -> np.signedinteger:
        pass


class MABThompsonSampling(MABPredictor):

    def __init__(self, mab: MABTrainer, s_multiplier: float = 1, f_multiplier: float = 1):
        super().__init__(mab)
        self.s_multiplier = s_multiplier
        self.f_multiplier = f_multiplier

    def select_arm(self, cluster_n: int) -> np.signedinteger:
        theta = np.zeros(len(self.mab.arms[cluster_n]))
        for arm_n in range(len(self.mab.arms[cluster_n])):
            theta[arm_n] = np.random.beta(self.mab.rewards[cluster_n, arm_n] * self.s_multiplier + 1,
                                          (1 - self.mab.rewards[cluster_n, arm_n]) * self.f_multiplier + 1)
        return np.argmax(theta)


class MABEpsilonGreedy(MABPredictor):

    def __init__(self, mab: MABTrainer, epsilon: float = 0.1):
        super().__init__(mab)
        self.epsilon = epsilon

    def select_arm(self, cluster_n: int) -> np.signedinteger:
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.mab.arms[cluster_n]))
        else:
            return np.argmax(self.mab.rewards[cluster_n])


def main(seed: int):
    random.seed(seed)

    dataset = datasets.CIC2017SubSample(load_preprocessed_data=False, seed=seed)

    for n_clusters in range(1, 15):
        logging.info(f"n_clusters: {n_clusters}")

        mab = MABTrainer(
            arms=[
                # Apollon
                models.RF(random_state=seed),
                models.DT(random_state=seed),
                models.NB(),
                # models.LR(),
                # models.MLP(),
                # My
                # models.GBC,
                # models.ET,
                # models.DF,
            ],
            n_clusters=n_clusters,
            random_state=seed,
            scorer=make_scorer(fbeta_score, beta=2)
        )

        mab.fit(
            X=dataset.X_train,
            y=dataset.y_train,
        )

        mab_predictor = MABThompsonSampling(mab)

        _, arms = mab_predictor.predict(dataset.X_test, return_arms=True)
        percentages = arms / np.sum(arms, axis=1)[:, np.newaxis]
        print(arms)
        print(np.round(percentages, 2))
        metrics.BasicMetrics(mab_predictor, dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test).report(only_test=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(
        seed=42
    )
