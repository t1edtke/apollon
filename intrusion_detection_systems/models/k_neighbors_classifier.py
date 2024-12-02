from sklearn.neighbors import KNeighborsClassifier


def CustomizedKNeighborsClassifier() -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=3,
        # n_jobs=-1,
    )
