from .metrics import Metrics


class BasicMetrics(Metrics):

    def report(self, only_training: bool = False, only_test: bool = True) -> None:
        print(f"Metrics for {self.model.__class__.__name__}")
        print("-" * 50)
        if not only_test:
            print(f"Train accuracy: {self.train_accuracy()}")
            print(f"Train precision: {self.train_precision()}")
            print(f"Train recall: {self.train_recall()}")
            print(f"Train f1: {self.train_f1()}")
            print(f"Train auc: {self.train_auc()}")
            print(f"Train confusion matrix:\n{self.train_confusion_matrix()}")
            print(f"Train classification report:\n{self.train_classification_report()}")
        if not (only_training or only_test):
            print()
        if not only_training:
            print(f"Test accuracy: {self.test_accuracy()}")
            print(f"Test precision: {self.test_precision()}")
            print(f"Test recall: {self.test_recall()}")
            print(f"Test f1: {self.test_f1()}")
            print(f"Test auc: {self.test_auc()}")
            print(f"Test confusion matrix:\n{self.test_confusion_matrix()}")
            print(f"Test classification report:\n{self.test_classification_report()}")
        print("-" * 50)
        print()
        print()
