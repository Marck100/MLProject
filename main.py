from Model.menu import MenuItem, Menu
from Model.analyzer import Analyzer
from Model.decision_tree_classifier import CDecisionTreeClassifier
from Model.nearest_neighbor_classifier import CNearestNeighborClassifier
from Model.svm_classifier import SVM
from Model.custom_classifier import CustomClassifier
from Model.multiple_classifier import MultipleClassifier
from Model.comparator import Comparator

# Show menu for dataset selection
dataset_item = MenuItem(
    'Default dataset',
    'Original dataset without any updates and changes',
    lambda: None
)

reduced_dataset_item = MenuItem(
    'Reduced dataset',
    'Smaller dataset (faster fits and operations - testing purposes)',
    lambda: None
)

dataset_paths = ['Resources/dataset.csv', 'Resources/reduced_dataset.csv']

dataset_menu = Menu(
    [
        dataset_item,
        reduced_dataset_item
    ]
)

dataset_menu.show()
choice = dataset_menu.askForChoice()

if choice == -1:
    exit(0)

# Selected dataset
dataset_path = dataset_paths[choice]

# Initialize analyzer (used for showing stats) + load
analyzer = Analyzer(dataset_path)
analyzer.load()
# Analyzed with balanced dataset
balanced_analyzer = Analyzer(dataset_path, balanced=True)
balanced_analyzer.load()

# Decision Tree Classifier
decision_tree_classifier = CDecisionTreeClassifier(dataset_path)
sd_decision_tree_classifier = CDecisionTreeClassifier(dataset_path, standardized=True)
fs_decision_tree_classifier = CDecisionTreeClassifier(dataset_path, feature_selection=True)
agg_decision_tree_classifier = CDecisionTreeClassifier(dataset_path, feature_aggregation=True)
bd_decision_tree_classifier = CDecisionTreeClassifier(dataset_path, balanced=True)
all_decision_tree_classifier = CDecisionTreeClassifier(
    dataset_path,
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# kNN
nearest_neighbor_classifier = CNearestNeighborClassifier(dataset_path)
sd_nearest_neighbor_classifier = CNearestNeighborClassifier(dataset_path, standardized=True)
fs_nearest_neighbor_classifier = CNearestNeighborClassifier(dataset_path, feature_selection=True)
agg_nearest_neighbor_classifier = CNearestNeighborClassifier(dataset_path, feature_aggregation=True)
bd_nearest_neighbor_classifier = CNearestNeighborClassifier(dataset_path, balanced=True)
all_nearest_neighbor_classifier = CNearestNeighborClassifier(
    dataset_path,
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# SVM
svm_classifier = SVM(dataset_path)
sd_svm_classifier = SVM(dataset_path, standardized=True)
fs_svm_classifier = SVM(dataset_path, feature_selection=True)
agg_svm_classifier = SVM(dataset_path, feature_aggregation=True)
bd_svm_classifier = SVM(dataset_path, balanced=True)
all_svm_classifier = SVM(
    dataset_path,
    standardized=False,
    feature_selection=False,
    feature_aggregation=False,
    balanced=True
)

# Custom
custom_classifier = CustomClassifier(dataset_path)
sd_custom_classifier = CustomClassifier(dataset_path, standardized=True)
fs_custom_classifier = CustomClassifier(dataset_path, feature_selection=True)
agg_custom_classifier = CustomClassifier(dataset_path, feature_aggregation=True)
bd_custom_classifier = CustomClassifier(dataset_path, balanced=True)
all_custom_classifier = CustomClassifier(
    dataset_path,
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# Multiple (kNN + decision tree)
multiple_classifier = MultipleClassifier(dataset_path)
sd_multiple_classifier = MultipleClassifier(dataset_path, standardized=True)
fs_multiple_classifier = MultipleClassifier(dataset_path, feature_selection=True)
agg_multiple_classifier = MultipleClassifier(dataset_path, feature_aggregation=True)
bd_multiple_classifier = MultipleClassifier(dataset_path, balanced=True)
all_multiple_classifier = MultipleClassifier(
    dataset_path,
    standardized=False,
    feature_selection=True,
    feature_aggregation=True,
    balanced=True
)

# Classifiers divided by preprocessing methods
# No preprocessing
classifiers_group = [
    decision_tree_classifier,
    nearest_neighbor_classifier,
    svm_classifier,
    custom_classifier,
    multiple_classifier
]
# Standardization
sd_classifiers_group = [
    sd_decision_tree_classifier,
    sd_nearest_neighbor_classifier,
    sd_svm_classifier,
    sd_custom_classifier,
    sd_multiple_classifier
]
# Feature selection
fs_classifiers_group = [
    fs_decision_tree_classifier,
    fs_nearest_neighbor_classifier,
    fs_svm_classifier,
    fs_custom_classifier,
    fs_multiple_classifier
]
# Feature aggregation
agg_classifiers_group = [
    agg_decision_tree_classifier,
    agg_nearest_neighbor_classifier,
    agg_svm_classifier,
    agg_custom_classifier,
    agg_multiple_classifier
]
# Balanced datasets
bd_classifiers_group = [
    bd_decision_tree_classifier,
    bd_nearest_neighbor_classifier,
    bd_svm_classifier,
    bd_custom_classifier,
    bd_multiple_classifier
]

# All methods without standardization
all_classifiers_group = [
    all_decision_tree_classifier,
    all_nearest_neighbor_classifier,
    all_svm_classifier,
    all_custom_classifier,
    all_multiple_classifier
]

classifiers_dict = {
    'Classifiers with no preprocessing': classifiers_group,
    'Classifiers with standardization': sd_classifiers_group,
    'Classifiers with feature selection': fs_classifiers_group,
    'Classifiers with feature aggregation': agg_classifiers_group,
    'Classifiers with balanced dataset': bd_classifiers_group,
    'Classifiers with all the previous pre processing techniques (no standardization)': all_classifiers_group
}

# Compare classifiers
comparator = Comparator(classifiers_dict)

# Menu initialization (items + menu)
# Option 1
stats_menu_item = MenuItem(
    'Stats',
    'Show stats (number of records and columns, duplicates, redundant elements)',
    lambda: [
        analyzer.showStats(),
        balanced_analyzer.showStats()
    ]
)

# Option 2
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

# Option 3
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

# Option 4
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

# Option 5
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

# Option 6
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

# Option 7
comparator_item = MenuItem(
    'Compare classifiers',
    'Compare accuracies (validation set) on different classifiers',
    lambda: [
        comparator.compareClassifiers()
    ]
)


menu = Menu(
    [
        stats_menu_item,
        decision_tree_classifier_item,
        nearest_neighbor_classifier_item,
        svm_classifier_item,
        custom_classifier_item,
        multiple_classifier_item,
        comparator_item
    ])

# Show menu until program ends
while 1:
    menu.show()
    if menu.askForChoice() == -1:
        break
    print()
