import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_PATH, 'air_quality_cleaned.csv'), parse_dates=['date'])
    return df

def plot_pollution_trends(target='PM2.5'):
    """
    Plot time-series trend of a pollution target across cities
    """
    df = load_data()
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x='date', y=target, hue='City', marker='o')
    plt.title(f"{target} Trends Over Time by City")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend(title='City')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix():
    """
    Plot correlation matrix of pollution features
    """
    df = load_data()
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Pollution Features")
    plt.tight_layout()
    plt.show()
