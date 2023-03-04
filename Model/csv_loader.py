import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA


class CSVLoader:

    _file_path: str
    _df = None
    _standardized = False
    _feature_selection = False
    _feature_aggregation = False

    def __init__(self, file_path, standardized=False, feature_selection=False, feature_aggregation=False):
        self._file_path = file_path

        self._standardized = standardized
        self._feature_selection = feature_selection
        self._feature_aggregation = feature_aggregation

    def load(self):
        self._df = pd.read_csv(self._file_path, encoding='utf8', sep=';')
        df = self._df

        if self._standardized:
            scaler = StandardScaler()

            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            class_column = df.columns[-1]

            df[class_column] = LabelEncoder().fit_transform(df[class_column])
            self._df = df

        if self._feature_selection:
            selector = SelectKBest(chi2, k=10)

            columns = df.columns

            x_s = df[columns[:-1]]

            y_s = df[columns[-1]]

            _ = selector.fit_transform(x_s, y_s)

            indexes = []
            for i, value in enumerate(selector.get_support()):
                if value:
                    indexes += [i]

            indexes += [len(columns) - 1]
            selected_columns = df.columns[indexes]

            data = df[selected_columns]

            self._df = data

        if self._feature_aggregation:
            n_components = 8
            pca = PCA(n_components)

            columns = df.columns

            x_s = df[columns[:-1]]

            f_columns = list(range(n_components))

            x_reduced = pca.fit_transform(x_s)

            self._df = pd.DataFrame(data=x_reduced, columns=f_columns)
            self._df[f_columns[-1]] = df[columns[-1]]


