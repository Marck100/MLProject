from Model.csv_loader import CSVLoader


class Analyzer(CSVLoader):

    def showStats(self):
        df = self._df
        records = df.shape[0]
        columns = len(df.columns)

        print('------Stats------')
        print(f'{records} records')
        print(f'{columns} columns')
        self.calcNone(display=True)
        self.calcRedundant(display=True)
        print('-----------------')


    def calcNone(self, display=False):
        df = self._df
        result = df.isnull().sum()
        if display:
            for index, item in enumerate(result):
                if item > 0:
                    print(f'{item} None fields for column {df.columns.values[index]}')

        return result

    def calcRedundant(self, display=False):
        df = self._df
        no_duplicates = df.drop_duplicates(inplace=False)
        diff = df.size - no_duplicates.size

        if display:
            print(f'{diff} duplicated records')

        return diff


if __name__ == '__main__':
    analyzer = Analyzer('../Resources/Customers.csv')
    analyzer.load()

    analyzer.showStats()