from typing import Any

from deepforest import CascadeForestClassifier  # installed from https://github.com/tien2020le2020/Deep-Forest


def CustomizedCascadeForestClassifier(random_state: Any) -> CascadeForestClassifier:
    return CascadeForestClassifier(
        # n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
