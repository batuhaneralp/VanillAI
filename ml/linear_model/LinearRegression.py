from common.classifier import Classifier


class LinearRegression(Classifier):
    def fit(self, X_train, y_train):
        return super().fit(X_train, y_train)

    def predict(self, X):
        return super().predict(X)

    def eval(self, X_train, y_train):
        return super().eval(X_train, y_train)
