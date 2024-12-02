from typing import Any

from sklearn.neural_network import MLPClassifier


def CustomizedMLPClassifier(random_state: Any) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='tanh',
        batch_size=200,
        random_state=random_state,
        early_stopping=True,
    )
