from typing import Any

from sklearn.linear_model import LogisticRegression


def CustomizedLogisticRegression(random_state: Any) -> LogisticRegression:
    return LogisticRegression(
        random_state=random_state,
        # n_jobs=-1,
    )
