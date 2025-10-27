import pandas as pd
import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', cfg['paths']['raw_data'])
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', cfg['paths']['processed_data'])


POLLUTION_COLS = cfg['features']['pollution']

def load_csv(filename):
    path = os.path.join(RAW_PATH, filename)
    df = pd.read_csv(path)
    return df

def clean_air_quality(df):
   
    df.rename(columns={'Date': 'date'}, inplace=True)

    cols_to_keep = ['City', 'date'] + [col for col in POLLUTION_COLS if col in df.columns]
    df = df[cols_to_keep]

    df['date'] = pd.to_datetime(df['date'])

    df.fillna(method='ffill', inplace=True)

    return df

def main():
    air_df = load_csv('air_quality_raw.csv')
    air_df = clean_air_quality(air_df)

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    air_df.to_csv(os.path.join(PROCESSED_PATH, 'air_quality_cleaned.csv'), index=False)
    print("Data preprocessing complete! Saved to processed folder.")

if __name__ == "__main__":
    main()
