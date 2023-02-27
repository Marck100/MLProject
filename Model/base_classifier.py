from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
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
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=random_state, test_size=test_size)

        return train_x, test_x, train_y, test_y

    def initClassifier(self):
        pass

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
    def accuracy(self, pred_y, test_y):
        correct = len(pred_y[pred_y == test_y])

        return correct/len(pred_y)

    # Wrong/Total
    def error_rate(self, pred_y, test_y):
        wrong = len(pred_y[pred_y != test_y])

        return wrong / len(pred_y)

    def showConfusionMatrix(self):
        _, _, _, test_y = self.splitSet()
        prediction = self.predict()

        matrix = confusion_matrix(test_y, prediction)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix)
        display.plot()
        plt.show()

    def showRocCurve(self):
        _, test_x, _, _ = self.splitSet()
        pred_y = self.predict()
        pred_prob_y = self._classifier.predict_proba(test_x)

        fpr, tpr, _ = roc_curve(pred_y, pred_prob_y[:, 1])

        plt.plot(fpr, tpr)
        plt.show()





if __name__ == '__main__':
    classifier = BaseClassifier('../Resources/Customers.csv')
    classifier.load()

    train_x, test_x, train_y, test_y = classifier.splitSet()
    print(train_x.shape, test_x.shape)

