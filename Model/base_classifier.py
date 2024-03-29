# Needed imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from Model.csv_loader import CSVLoader
import matplotlib.pyplot as plt


# Inherit CSVLoader
class BaseClassifier(CSVLoader):
    _classifier = None
    _prepared = False

    # Split features and classes
    def prepareSet(self):
        df = self._df

        x_s = df.iloc[:, 0:-1].to_numpy()
        y_s = df.iloc[:, -1].to_numpy()

        return x_s, y_s

    def splitSet(self, random_state=0, test_size=0.25):
        x, y = self.prepareSet()
        # TODO Remove random_state (??)
        # Split dataset into validation and training set
        return train_test_split(x, y, random_state=random_state, test_size=test_size)

    # Init _classifier variable
    def initClassifier(self):
        pass

    # Prepare classifier (load dataset, initialize classifier, fit classifier)
    def prepare(self):
        if not self._prepared:
            self.load()
            self.initClassifier()
            self.fitClassifier()

            self._prepared = True

    # Show classifier parameters
    def showParameters(self):
        print(self._classifier.get_params())

    # Return better classifier (based on parameters tuning)
    def tuneParameters(self):
        pass

    # Fit classifier with training set
    def fitClassifier(self):
        train_x, _, train_y, _ = self.splitSet()
        self._classifier.fit(train_x, train_y)

    # Predict labels using validation set
    def predict(self):
        _, test_x, _, _ = self.splitSet()
        pred_y = self._classifier.predict(test_x)

        return pred_y

    def score(self, x, y):
        return self._classifier.score(x, y)

    # Correct/Total
    @staticmethod
    def accuracy(pred_y, test_y):
        correct = len(pred_y[pred_y == test_y])

        return correct / len(pred_y)

    # Wrong/Total
    @staticmethod
    def error_rate(pred_y, test_y):
        wrong = len(pred_y[pred_y != test_y])

        return wrong / len(pred_y)

    # Display confusion matrix
    def showConfusionMatrix(self):
        _, _, _, test_y = self.splitSet()
        prediction = self.predict()

        matrix = confusion_matrix(test_y, prediction)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix)
        display.plot()
        plt.show()

    # Show validation stats
    def validationResult(self, display_label=''):
        prediction = self.predict()

        train_x, test_x, train_y, test_y = self.splitSet()

        print()
        print(display_label)

        print(f'Training set score: {self.score(train_x, train_y)}')

        print(f'Validation set accuracy: {self.accuracy(prediction, test_y)}')
        print(f'Validation set error rate: {self.error_rate(prediction, test_y)}')

        self.showConfusionMatrix()

    # Compute accuracy on validation set (see comparator.py)
    def validationSetAccuracy(self):
        prediction = self.predict()
        _, _, _, test_y = self.splitSet()

        return self.accuracy(prediction, test_y)


if __name__ == '__main__':
    classifier = BaseClassifier('../Resources/dataset.csv')
    classifier.load()

    print(list(map(lambda x: x.size, classifier.splitSet())))
