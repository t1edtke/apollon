from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier

from .attack import Attack


class ZerothOffsetOptimizationAttack(Attack):

    def get_attack(self, classifier: SklearnClassifier):
        return ZooAttack(classifier=classifier, targeted=True, nb_parallel=64)


if __name__ == '__main__':
    from intrusion_detection_systems import models
    import datasets

    model = models.RF(random_state=42)
    dataset = datasets.CIC2017SubSample(load_preprocessed_data=False, seed=42)
    model.fit(X=dataset.X_train, y=dataset.y_train)

    X_adv = ZerothOffsetOptimizationAttack().generate_adversarial_examples(
        model=model,
        dataset=dataset,
    )
    print(X_adv)
    print(X_adv.shape)
    print(X_adv[0])
    print(X_adv[0].shape)
