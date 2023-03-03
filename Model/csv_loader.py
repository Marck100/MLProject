import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CSVLoader:

    _file_path: str
    _df = None
    _standardized = False

    def __init__(self, file_path, standardized=False):
        self._file_path = file_path

        self._standardized = standardized

    def load(self):
        self._df = pd.read_csv(self._file_path, encoding='utf8', sep=';')
        df = self._df

        if self._standardized:
            scaler = StandardScaler()

            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            class_column = df.columns[-1]

            df[class_column] = LabelEncoder().fit_transform(df[class_column])
            self._df = df


