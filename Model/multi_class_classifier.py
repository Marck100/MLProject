# Needed imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from Model.base_classifier import BaseClassifier
from sklearn.ensemble import VotingClassifier


# Inherit BaseClassifier
class MultiClassClassifier(BaseClassifier):

    def initClassifier(self):
        # Uses nearest neighbor and decision tree
        k_nn = KNeighborsClassifier(1)
        d_tree = DecisionTreeClassifier()

        self._classifier = VotingClassifier(estimators=[('knn', k_nn), ('tree', d_tree)])

    def tuneParameters(self):
        pass


if __name__ == '__main__':
    classifier = MultiClassClassifier('../Resources/dataset.csv')
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
