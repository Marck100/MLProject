# Needed imports
from sklearn.svm import SVC
from Model.base_classifier import BaseClassifier
from sklearn.model_selection import GridSearchCV


# Inherit BaseClassifier
class SVM(BaseClassifier):

    def initClassifier(self):
        self._classifier = self.tuneParameters()

    # Return the best classifier
    def tuneParameters(self):
        train_x, _, train_y, _ = self.splitSet()
        # Params for tuning
        params = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
        }

        # Tuning
        search = GridSearchCV(SVC(), params, scoring='accuracy')
        search.fit(train_x, train_y)

        best_params = search.best_params_

        # Return classifier with best parameters
        return SVC(**best_params)


if __name__ == '__main__':
    classifier = SVM('../Resources/dataset.csv')
    classifier.load()
    classifier.initClassifier()
    classifier.showParameters()
    classifier.fitClassifier()
    prediction = classifier.predict()
    print(prediction)

    trainX, testX, trainY, testY = classifier.splitSet()
    training_set_score = classifier.score(trainX, trainY)
    validation_set_score = classifier.score(testX, testY)

    training_set_accuracy = classifier.score(trainX, trainY)
    validation_set_accuracy = classifier.score(testX, testY)

    print(f'Validation set score: {validation_set_score}')
    print(f'Training set score: {training_set_score}')

    print(f'Validation set accuracy: {classifier.accuracy(prediction, testY)}')
    print(f'Validation set error rate: {classifier.error_rate(prediction, testY)}')

    classifier.showConfusionMatrix()
