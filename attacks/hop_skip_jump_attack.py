from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

from .attack import Attack


class HopSkipJumpAttack(Attack):

    def get_attack(self, classifier: SklearnClassifier):
        return HopSkipJump(classifier=classifier, targeted=True)


if __name__ == '__main__':
    from intrusion_detection_systems import models
    import datasets

    model = models.RF(random_state=42)
    dataset = datasets.CIC2017SubSample(load_preprocessed_data=False, seed=42)
    model.fit(X=dataset.X_train, y=dataset.y_train)

    X_adv = HopSkipJumpAttack().generate_adversarial_examples(
        model=model,
        dataset=dataset,
    )
    print(X_adv)
    print(X_adv.shape)
    print(X_adv[0])
    print(X_adv[0].shape)
