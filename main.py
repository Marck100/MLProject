from Model.menu import MenuItem, Menu
from Model.analyzer import Analyzer
from Model.decision_tree_classifier import CDecisionTreeClassifier
from Model.nearest_neighbor_classifier import CNearestNeighborClassifier
from Model.custom_classifier import CustomClassifier
from Model.multi_class_classifier import MultiClassClassifier

analyzer = Analyzer('Resources/dataset.csv')
analyzer.load()

decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv')
decision_tree_classifier.prepare()

sd_decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv', standardized=True)
sd_decision_tree_classifier.prepare()

nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv')
nearest_neighbor_classifier.prepare()

sd_nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv', standardized=True)
sd_nearest_neighbor_classifier.prepare()

custom_classifier = CustomClassifier('Resources/dataset.csv')
custom_classifier.prepare()

sd_custom_classifier = CustomClassifier('Resources/dataset.csv', standardized=True)
sd_custom_classifier.prepare()

multiple_classifier = MultiClassClassifier('Resources/dataset.csv')
multiple_classifier.prepare()

sd_multiple_classifier = MultiClassClassifier('Resources/dataset.csv', standardized=True)
sd_multiple_classifier.prepare()

stats_menu_item = MenuItem(
    'Stats',
    'Show stats (number of records and columns, duplicates, redundant elements)',
    lambda: analyzer.showStats()
)

decision_tree_classifier_item = MenuItem(
    'Decision tree classifier',
    'Validate prediction on a decision tree classifier',
    lambda: [
        decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats'
        ),
        sd_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with standardized dataset'
        )
    ]
)

nearest_neighbor_classifier_item = MenuItem(
    'Nearest neighbor classifier',
    'Validate prediction on a nearest neighbor classifier',
    lambda: [
        nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats'
        ),
        sd_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with standardized dataset'
        )
    ]
)

custom_classifier_item = MenuItem(
    'Custom classifier',
    'Validate prediction on a custom classifier',
    lambda: [
        custom_classifier.validationResult(
            display_label='Custom Classifier - Stats'
        ),
        sd_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with standardized dataset'
        )
    ]
)

multiple_classifier_item = MenuItem(
    'Multiple classifier',
    'Validate prediction on a multiple classifier',
    lambda: [
        multiple_classifier.validationResult(
            display_label='Multiple Classifier - Stats'
        ),
        sd_multiple_classifier.validationResult(
            display_label='Multiple Classifier - Stats with standardized dataset'
        )
    ]
)


menu = Menu(
    [
        stats_menu_item,
        decision_tree_classifier_item,
        nearest_neighbor_classifier_item,
        custom_classifier_item,
        multiple_classifier_item
    ])

while 1:
    menu.show()
    if menu.askForChoice() == -1:
        break
    print()
