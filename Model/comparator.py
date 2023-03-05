from Model.base_classifier import BaseClassifier
from Model.decision_tree_classifier import CDecisionTreeClassifier
from Model.nearest_neighbor_classifier import CNearestNeighborClassifier
import matplotlib.pyplot as plt


class Comparator:

    _classifiers: {str: [BaseClassifier]}

    def __init__(self, classifiers: {str: [BaseClassifier]}):
        self._classifiers = classifiers

    def compareClassifiers(self):
        classifiers = self._classifiers
        n_groups = len(classifiers)
        fig, axs = plt.subplots(n_groups)

        for index, key in enumerate(classifiers):
            classifier_group = classifiers[key]
            for classifier in classifier_group:
                classifier.prepare()

            classifiers_name = list(map(lambda x: x.__class__.__name__, classifier_group))

            validation_set_accuracies = list(map(lambda x: x.validationSetAccuracy(), classifier_group))

            axs[index].barh(classifiers_name, validation_set_accuracies)
            axs[index].title.set_text(key)

        fig.suptitle('Validation set scores')

        #fig.set_size_inches(12, 20, forward=True)
        fig.set_size_inches(20, 20, forward=True)

        plt.subplots_adjust(hspace=1)
        plt.show()


if __name__ == '__main__':
    decision_tree_classifier = CDecisionTreeClassifier('../Resources/dataset.csv')
    nearest_neighbor_classifier = CNearestNeighborClassifier('../Resources/dataset.csv')

    all_decision_tree_classifier = CDecisionTreeClassifier(
        '../Resources/dataset.csv',
        standardized=False,
        feature_selection=True,
        feature_aggregation=True,
        balanced=True
    )
    all_nearest_neighbor_classifier = CNearestNeighborClassifier(
        '../Resources/dataset.csv',
        standardized=False,
        feature_selection=True,
        feature_aggregation=True,
        balanced=True
    )

    classifiers_group = [decision_tree_classifier, nearest_neighbor_classifier]
    all_classifiers_group = [all_decision_tree_classifier, all_nearest_neighbor_classifier]

    comparator = Comparator({
        'no preprocessing': classifiers_group,
        'pre processing': all_classifiers_group
    })

    comparator.compareClassifiers()
