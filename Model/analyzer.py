import numpy

from Model.csv_loader import CSVLoader


class Analyzer(CSVLoader):

    def showStats(self):
        # Dataframe loaded with pandas
        df = self._df
        records = df.shape[0]
        columns = len(df.columns)

        # Show stats (dataset quality and indexes)
        print('------Stats------')
        print()
        print(df.describe().to_string())
        print()
        print(f'{records} records')
        print(f'{columns} columns')
        self.calcNone(display=True)
        self.calcRedundant(display=True)
        print()

        print('\n')
        print(f'Outliers:')
        self.calcOutliers(display=True)
        print('-----------------')

    def calcNone(self, display=False):
        df = self._df
        # None values for each feature (column)
        result = df.isnull().sum()
        flag = 0

        # Print none value for each feature (just once if dataset doesn't contain any none value)
        if display:
            for index, item in enumerate(result):
                if item > 0:
                    print(f'{item} None fields for column {df.columns.values[index]}')
                    flag = 1

            if flag == 0:
                print('0 None values')

        return result

    def calcRedundant(self, display=False):
        df = self._df

        # redundant = all records - dataframe without duplicates
        no_duplicates = df.drop_duplicates(inplace=False, keep='first')
        diff = df.shape[0] - no_duplicates.shape[0]

        if display:
            print(f'{diff} duplicated records')

        return diff

    def calcOutliers(self, display=False):
        df = self._df
        # 1, 3 quartile
        quantiles = numpy.array(df.quantile([0.25, 0.75]))

        # X_i is outlier if one of the following condition is full filled
        # X_i < quartile - const * (3 quartile - 1 quartile)
        # X_i > 3 quartile + const * (3 quartile - 1 quartile)
        const = 1.5
        cond1 = quantiles[0] - const * (quantiles[1] - quantiles[0])
        cond2 = quantiles[1] + const * (quantiles[1] - quantiles[0])

        index = 0
        for col in df:
            if col == 'quality':
                break
            outliers = []
            for value in df[col]:
                if value < cond1[index] or value > cond2[index]:
                    outliers += [value]

            if display:
                print(f'{col} ({len(outliers)})')
                print(numpy.array(outliers))
                print()
            index += 1


if __name__ == '__main__':
    analyzer = Analyzer('../Resources/dataset.csv')
    analyzer.load()

    analyzer.showStats()
