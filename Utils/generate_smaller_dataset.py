import pandas as pd
from sklearn.utils import resample

# Create reduced_dataset
df = pd.read_csv('../Resources/dataset.csv', encoding='utf-8', sep=';')
new_df = resample(df, replace=False, n_samples=200)

# Save new dataset
new_df.to_csv('../Resources/reduced_dataset.csv', sep=';', index=None)
