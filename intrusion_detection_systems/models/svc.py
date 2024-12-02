from typing import Any

from sklearn.svm import SVC


def CustomizedSVC(random_state: Any) -> SVC:
    return SVC(
        random_state=random_state,
    )
