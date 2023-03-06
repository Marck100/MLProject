# Needed imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# Class that loads the dataset
class CSVLoader:

    # Dataset path
    _file_path: str
    _df = None
    # Standardized dataset
    _standardized = False
    # Feature selection
    _feature_selection = False
    # Feature aggregation
    _feature_aggregation = False
    # Balanced dataset
    _balanced = False

    # Initialize the loader
    def __init__(self, file_path, standardized=False, feature_selection=False, feature_aggregation=False, balanced=False):
        self._file_path = file_path
        self._standardized = standardized
        self._feature_selection = feature_selection
        self._feature_aggregation = feature_aggregation
        self._balanced = balanced

    # Read the dataset from the file path
    def load(self):
        self._df = pd.read_csv(self._file_path, encoding='utf8', sep=';')
        df = self._df

        if self._feature_selection:
            # Select the needed feature
            selector = SelectKBest(chi2, k=10)

            # Select the columns
            columns = df.columns

            # Records (no classes)
            x_s = df[columns[:-1]]

            # Classes
            y_s = df[columns[-1]]

            # Variable not needed (_)
            _ = selector.fit_transform(x_s, y_s)

            # Final columns
            selector_indexes = selector.get_support(indices=True)
            selected_columns = df.columns[selector_indexes].tolist()
            selected_columns += [columns[-1]]

            data = df[selected_columns]

            self._df = data

        if self._feature_aggregation:
            # Number of final components
            n_components = 8
            pca = PCA(n_components)

            columns = df.columns

            x_s = df[columns[:-1]]

            # New columns named by indexes
            f_columns = list(range(n_components))

            x_reduced = pca.fit_transform(x_s)

            # Create the dataframe
            self._df = pd.DataFrame(data=x_reduced, columns=f_columns)
            self._df[len(f_columns)] = df[columns[-1]]

        # Balance the dataset
        if self._balanced:
            over = RandomOverSampler()
            under = RandomUnderSampler()

            columns = df.columns

            x_s = df[columns[:-1]]
            y_s = df[columns[-1]]

            # Oversampling
            new_x, new_y = over.fit_resample(x_s, y_s)
            # Under-sampling
            new_x, new_y = under.fit_resample(new_x, new_y)

            data = new_x

            self._df = pd.DataFrame(data=data, columns=columns)
            self._df[columns[-1]] = new_y

        # Standardize the dataset
        if self._standardized:
            scaler = StandardScaler()

            # Transform records (fit_transform)
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            class_column = df.columns[-1]

            # LabelEncoder discretize classes
            df[class_column] = LabelEncoder().fit_transform(df[class_column])
            self._df = df

