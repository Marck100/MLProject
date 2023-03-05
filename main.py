from Model.menu import MenuItem, Menu
from Model.analyzer import Analyzer
from Model.decision_tree_classifier import CDecisionTreeClassifier
from Model.nearest_neighbor_classifier import CNearestNeighborClassifier
from Model.svm_classifier import SVM
from Model.custom_classifier import CustomClassifier
from Model.multi_class_classifier import MultiClassClassifier

analyzer = Analyzer('Resources/dataset.csv')
analyzer.load()

balanced_analyzer = Analyzer('Resources/dataset.csv', balanced=True)
balanced_analyzer.load()

# Decision Tree Classifier
decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv')
sd_decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv', standardized=True)
fs_decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv', feature_selection=True)
agg_decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv', feature_aggregation=True)
bd_decision_tree_classifier = CDecisionTreeClassifier('Resources/dataset.csv', balanced=True)
all_decision_tree_classifier = CDecisionTreeClassifier(
    'Resources/dataset.csv',
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# kNN
nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv')
sd_nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv', standardized=True)
fs_nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv', feature_selection=True)
agg_nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv', feature_aggregation=True)
bd_nearest_neighbor_classifier = CNearestNeighborClassifier('Resources/dataset.csv', balanced=True)
all_nearest_neighbor_classifier = CNearestNeighborClassifier(
    'Resources/dataset.csv',
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)
# SVM
svm_classifier = SVM('Resources/dataset.csv')
sd_svm_classifier = SVM('Resources/dataset.csv', standardized=True)
fs_svm_classifier = SVM('Resources/dataset.csv', feature_selection=True)
agg_svm_classifier = SVM('Resources/dataset.csv', feature_aggregation=True)
bd_svm_classifier = SVM('Resources/dataset.csv', balanced=True)
all_svm_classifier = SVM(
    'Resources/dataset.csv',
    standardized=False,
    feature_selection=False,
    feature_aggregation=False,
    balanced=True
)

# Custom
custom_classifier = CustomClassifier('Resources/dataset.csv')
sd_custom_classifier = CustomClassifier('Resources/dataset.csv', standardized=True)
fs_custom_classifier = CustomClassifier('Resources/dataset.csv', feature_selection=True)
agg_custom_classifier = CustomClassifier('Resources/dataset.csv', feature_aggregation=True)
bd_custom_classifier = CustomClassifier('Resources/dataset.csv', balanced=True)
all_custom_classifier = CustomClassifier(
    'Resources/dataset.csv',
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# Multiple
multiple_classifier = MultiClassClassifier('Resources/dataset.csv')
sd_multiple_classifier = MultiClassClassifier('Resources/dataset.csv', standardized=True)
fs_multiple_classifier = MultiClassClassifier('Resources/dataset.csv', feature_selection=True)
agg_multiple_classifier = MultiClassClassifier('Resources/dataset.csv', feature_aggregation=True)
bd_multiple_classifier = MultiClassClassifier('Resources/dataset.csv', balanced=True)
all_multiple_classifier = MultiClassClassifier(
    'Resources/dataset.csv',
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

stats_menu_item = MenuItem(
    'Stats',
    'Show stats (number of records and columns, duplicates, redundant elements)',
    lambda: [
        analyzer.showStats(),
        balanced_analyzer.showStats()
    ]
)

decision_tree_classifier_item = MenuItem(
    'Decision tree classifier',
    'Validate prediction on a decision tree classifier',
    lambda: [
        decision_tree_classifier.prepare(),
        sd_decision_tree_classifier.prepare(),
        fs_decision_tree_classifier.prepare(),
        agg_decision_tree_classifier.prepare(),
        bd_decision_tree_classifier.prepare(),
        all_decision_tree_classifier.prepare(),
        decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats'
        ),
        sd_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with standardized dataset'
        ),
        fs_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with feature selection'
        ),
        agg_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with feature aggregation'
        ),
        bd_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with balanced dataset'
        ),
        all_decision_tree_classifier.validationResult(
            display_label='Decision Tree Classifier - Stats with combined preprocessing'
        )
    ]
)

nearest_neighbor_classifier_item = MenuItem(
    'Nearest neighbor classifier',
    'Validate prediction on a nearest neighbor classifier',
    lambda: [
        nearest_neighbor_classifier.prepare(),
        sd_nearest_neighbor_classifier.prepare(),
        fs_nearest_neighbor_classifier.prepare(),
        agg_nearest_neighbor_classifier.prepare(),
        bd_nearest_neighbor_classifier.prepare(),
        all_nearest_neighbor_classifier.prepare(),
        nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats'
        ),
        sd_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with standardized dataset'
        ),
        fs_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with feature selection'
        ),
        agg_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with feature aggregation'
        ),
        bd_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with balanced dataset'
        ),
        all_nearest_neighbor_classifier.validationResult(
            display_label='Nearest Neighbor Classifier - Stats with combined preprocessing'
        )
    ]
)

svm_classifier_item = MenuItem(
    'SVM classifier',
    'Validate prediction on an SVM classifier',
    lambda: [
        svm_classifier.prepare(),
        sd_svm_classifier.prepare(),
        fs_svm_classifier.prepare(),
        agg_svm_classifier.prepare(),
        bd_svm_classifier.prepare(),
        all_svm_classifier.prepare(),
        svm_classifier.validationResult(
            display_label='SVM - Stats'
        ),
        sd_svm_classifier.validationResult(
            display_label='SVM - Stats with standardized dataset'
        ),
        fs_svm_classifier.validationResult(
            display_label='SVM - Stats with feature selection'
        ),
        agg_svm_classifier.validationResult(
            display_label='SVM - Stats with feature aggregation'
        ),
        bd_svm_classifier.validationResult(
            display_label='SVM - Stats with balanced dataset'
        ),
        all_svm_classifier.validationResult(
            display_label='SVM - Stats with combined preprocessing'
        )
    ]
)

custom_classifier_item = MenuItem(
    'Custom classifier',
    'Validate prediction on a custom classifier',
    lambda: [
        custom_classifier.prepare(),
        sd_custom_classifier.prepare(),
        fs_custom_classifier.prepare(),
        agg_custom_classifier.prepare(),
        bd_custom_classifier.prepare(),
        all_custom_classifier.prepare(),
        custom_classifier.validationResult(
            display_label='Custom Classifier - Stats'
        ),
        sd_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with standardized dataset'
        ),
        fs_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with feature selection'
        ),
        agg_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with feature aggregation'
        ),
        bd_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with balanced dataset'
        ),
        all_custom_classifier.validationResult(
            display_label='Custom Classifier - Stats with combined preprocessing'
        )
    ]
)

multiple_classifier_item = MenuItem(
    'Multiple classifier',
    'Validate prediction on a multiple classifier',
    lambda: [
        multiple_classifier.prepare(),
        sd_multiple_classifier.prepare(),
        fs_custom_classifier.prepare(),
        agg_custom_classifier.prepare(),
        bd_custom_classifier.prepare(),
        all_custom_classifier.prepare(),
        multiple_classifier.validationResult(
            display_label='Multiple Classifier - Stats'
        ),
        sd_multiple_classifier.validationResult(
            display_label='Multiple Classifier - Stats with standardized dataset'
        ),
        fs_custom_classifier.validationResult(
            display_label='Multiple Classifier - Stats with feature selection'
        ),
        agg_custom_classifier.validationResult(
            display_label='Multiple Classifier - Stats with feature aggregation'
        ),
        bd_custom_classifier.validationResult(
            display_label='Multiple Classifier - Stats with balanced dataset'
        ),
        all_custom_classifier.validationResult(
            display_label='Multiple Classifier - Stats with combined preprocessing'
        )
    ]
)


menu = Menu(
    [
        stats_menu_item,
        decision_tree_classifier_item,
        nearest_neighbor_classifier_item,
        svm_classifier_item,
        custom_classifier_item,
        multiple_classifier_item
    ])

while 1:
    menu.show()
    if menu.askForChoice() == -1:
        break
    print()
