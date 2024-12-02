from typing import Any

from sklearn.ensemble import GradientBoostingClassifier


def CustomizedGradientBoostingClassifier(random_state: Any) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        random_state=random_state,
    )
