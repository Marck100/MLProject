from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from Model.csv_loader import CSVLoader
import matplotlib.pyplot as plt


class BaseClassifier(CSVLoader):
    _classifier = None

    # Split features from classes
    def prepareSet(self):
        df = self._df

        x_s = df.iloc[:, 0:-1].to_numpy()
        y_s = df.iloc[:, -1].to_numpy()

        return x_s, y_s

    def splitSet(self, random_state=0, test_size=0.25):
        x, y = self.prepareSet()
        return train_test_split(x, y, random_state=random_state, test_size=test_size)

    def initClassifier(self):
        pass

    def prepare(self):
        self.load()
        self.initClassifier()
        self.fitClassifier()

    def showParameters(self):
        print(self._classifier.get_params())

    def tuneParameters(self):
        pass

    def fitClassifier(self):
        train_x, _, train_y, _ = self.splitSet()
        self._classifier.fit(train_x, train_y)

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

    def showConfusionMatrix(self):
        _, _, _, test_y = self.splitSet()
        prediction = self.predict()

        matrix = confusion_matrix(test_y, prediction)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix)
        display.plot()
        plt.show()

    # Show validation stats
    def validationResult(self, display_label=''):
        self.load()
        self.initClassifier()
        self.fitClassifier()
        prediction = self.predict()

        train_x, test_x, train_y, test_y = self.splitSet()

        print()
        print(display_label)

        print(f'Validation set score: {self.score(test_x, test_y)}')
        print(f'Training set score: {self.score(train_x, train_y)}')

        print(f'Validation set accuracy: {self.accuracy(prediction, test_y)}')
        print(f'Validation set error rate: {self.error_rate(prediction, test_y)}')

        self.showConfusionMatrix()


if __name__ == '__main__':
    classifier = BaseClassifier('../Resources/dataset.csv')
    classifier.load()

    print(list(map(lambda x: x.size, classifier.splitSet())))
