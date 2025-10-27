import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_PATH, 'air_quality_cleaned.csv'), parse_dates=['date'])
    return df

def basic_stats():
    df = load_data()
    print("First 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def plot_distributions():
    df = load_data()
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_time_series(target='PM2.5'):
    df = load_data()
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x='date', y=target, hue='City', marker='o')
    plt.title(f'{target} Over Time by City')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()
