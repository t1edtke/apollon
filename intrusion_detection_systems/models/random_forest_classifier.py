from typing import Any

from sklearn.ensemble import RandomForestClassifier


def CustomizedRandomForestClassifier(random_state: Any) -> RandomForestClassifier:
    return RandomForestClassifier(
        # n_jobs=-1,
        random_state=random_state,
    )
