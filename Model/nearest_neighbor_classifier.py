from sklearn.neighbors import KNeighborsClassifier
from Model.base_classifier import BaseClassifier
from sklearn.model_selection import GridSearchCV


class CNearestNeighborClassifier(BaseClassifier):

    def initClassifier(self):
        self._classifier = self.tuneParameters()

    def tuneParameters(self):
        train_x, _, train_y, _ = self.splitSet()
        params = {
            'n_neighbors': list(range(1, 20)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }

        search = GridSearchCV(KNeighborsClassifier(), params, scoring='accuracy')
        search.fit(train_x, train_y)

        best_params = search.best_params_

        return KNeighborsClassifier(**best_params)


if __name__ == '__main__':
    classifier = CNearestNeighborClassifier('../Resources/dataset.csv')
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
