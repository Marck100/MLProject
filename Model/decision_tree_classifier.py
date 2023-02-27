from sklearn.tree import DecisionTreeClassifier
from Model.base_classifier import BaseClassifier


class CDecisionTreeClassifier(BaseClassifier):

    def initClassifier(self):
        self._classifier = self.tuneParameters()

    def tuneParameters(self):
        depths = list(range(1, 11))

        train_x, test_x, train_y, test_y = classifier.splitSet()
        classifiers = list(map(lambda x: DecisionTreeClassifier(max_depth=x, random_state=0), depths))
        for cls in classifiers:
            cls.fit(train_x, train_y)

        return max(classifiers, key= lambda x: x.score(test_x, test_y))



if __name__ == '__main__':
    classifier = CDecisionTreeClassifier('../Resources/Customers.csv')
    classifier.load()
    classifier.initClassifier()
    classifier.showParameters()
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