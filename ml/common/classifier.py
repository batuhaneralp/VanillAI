from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def eval(self, X_train, y_train):
        pass
