from sklearn.dummy import DummyClassifier
from Model.base_classifier import BaseClassifier


class CDummyClassifier(BaseClassifier):

    def initClassifier(self):
        self._classifier = DummyClassifier(strategy='stratified')


if __name__ == '__main__':
    classifier = CDummyClassifier('../Resources/dataset.csv')
    classifier.load()
    classifier.initClassifier()
    classifier.fitClassifier()
    prediction = classifier.predict()
    print(prediction)

    train_x, test_x, train_y, test_y = classifier.splitSet()
    training_set_score = classifier.score(train_x, train_y)
    validation_set_score = classifier.score(test_x, test_y)

    training_set_accuracy = classifier.score(train_x, train_y)
    validation_set_accuracy = classifier.score(test_x, test_y)

    print(f'Validation set score: {validation_set_score}')
    print(f'Training set score: {training_set_score}')

    print(f'Validation set accuracy: {classifier.accuracy(prediction, test_y)}')
    print(f'Validation set error rate: {classifier.error_rate(prediction, test_y)}')

    classifier.showConfusionMatrix()
    classifier.showRocCurve()
