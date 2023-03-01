from Model.base_classifier import BaseClassifier


class CustomClassifier(BaseClassifier):
    _l_s = []

    def initClassifier(self):
        pass

    def fitClassifier(self):
        pass


if __name__ == '__main__':
    classifier = CustomClassifier('../Resources/dataset.csv')
    classifier.load()
    classifier.initClassifier()
