from typing import Any

from sklearn.ensemble import ExtraTreesClassifier


def CustomizedExtraTreesClassifier(random_state: Any) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        # n_jobs=-1,
        random_state=random_state,
    )
