import pandas as pd

# Add a binary class (trusted) based on spending score (only works for old_dataset.csv)
if __name__ == '__main__':
    df = pd.read_csv('../Resources/dataset.csv')
    print(df['Spending Score (1-100)'])
    df['Trusted'] = (df['Spending Score (1-100)'] >= 70).astype(int)
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df.drop('Profession', axis=1, inplace=True)
    df.to_csv('../Resources/dataset.csv', index=False)

    print(df.to_string())
