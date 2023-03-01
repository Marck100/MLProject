from Model.base_classifier import BaseClassifier
import numpy as np


class CustomClassifier(BaseClassifier):
    _l_s = []

    def initClassifier(self):
        pass

    def fitClassifier(self):
        pass

    def showParameters(self):
        pass

    # Custom kNN (calc differences between your record and the dataset using mean)
    def predict(self):
        train_x, test_x, train_y, test_y = self.splitSet()

        def predict(X):
            difference = []
            for element in train_x:
                differences = np.asarray(abs(element - X))
                difference += [differences.mean()]

            result = train_y[difference.index(min(difference))]
            return result

        return np.asarray(list(map(lambda x: predict(x), test_x)))

    def score(self, i_x, i_y):

        def predict(X):
            difference = []
            for element in i_x:
                differences = np.asarray(abs(element - X))
                difference += [differences.mean()]

            result = i_y[difference.index(min(difference))]
            return result

        predictions = np.asarray(list(map(lambda x: predict(x), i_x)))

        return self.accuracy(predictions, i_y)

    # Correct/Total
    @staticmethod
    def accuracy(pred_y, test_y):
        correct = len(pred_y[pred_y == test_y])

        return correct / len(pred_y)

    @staticmethod
    def error_rate(pred_y, test_y):
        wrong = len(pred_y[pred_y != test_y])

        return wrong / len(pred_y)


if __name__ == '__main__':
    classifier = CustomClassifier('../Resources/dataset.csv')
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
    classifier.showRocCurve()
