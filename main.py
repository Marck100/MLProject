from Model.menu import MenuItem, Menu
from Model.analyzer import Analyzer
from Model.dummy_classifier import CDummyClassifier
from Model.decision_tree_classifier import CDecisionTreeClassifier
from Model.nearest_neighbor_classifier import CNearestNeighborClassifier

analyzer = Analyzer('Resources/dataset.csv')
analyzer.load()

dummy_classifier = CDummyClassifier('Resources/dataset.csv')
dummy_classifier.prepare()

decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv')
decision_tree_classifier.prepare()

nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv')
nearest_neighbor_classifier.prepare()

stats_menu_item = MenuItem(
    'Stats',
    'Show stats (number of records and columns, duplicates, redundant elements)',
    lambda: analyzer.showStats()
)

dummy_classifier_item = MenuItem(
    'Dummy classifier',
    'Validate prediction on a dummy classifier',
    lambda: dummy_classifier.validationResult()
)

decision_tree_classifier_item = MenuItem(
    'Decision tree classifier',
    'Validate prediction on a decision tree classifier',
    lambda: decision_tree_classifier.validationResult()
)

nearest_neighbor_classifier_item = MenuItem(
    'Nearest neighbor classifier',
    'Validate prediction on a nearest neighbor classifier',
    lambda: nearest_neighbor_classifier.validationResult()
)

menu = Menu([stats_menu_item, dummy_classifier_item, decision_tree_classifier_item, nearest_neighbor_classifier_item])

while 1:
    menu.show()
    if menu.askForChoice() == -1:
        break
    print()
