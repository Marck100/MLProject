import pandas as pd
from sklearn.utils import resample


df = pd.read_csv('../Resources/dataset.csv', encoding='utf-8', sep=';')
new_df = resample(df, replace=False, n_samples=200)

new_df.to_csv('../Resources/reduced_dataset.csv', sep=';', index=None)
