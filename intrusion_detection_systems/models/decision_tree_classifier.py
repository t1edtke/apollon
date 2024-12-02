from typing import Any

from sklearn.tree import DecisionTreeClassifier


def CustomizedDecisionTreeClassifier(random_state: Any) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        random_state=random_state,
    )
