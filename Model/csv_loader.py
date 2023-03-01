import pandas as pd


class CSVLoader:

    _file_path: str
    _df = None

    def __init__(self, file_path):
        self._file_path = file_path

    def load(self):
        self._df = pd.read_csv(self._file_path, encoding='utf8', sep=';')
