import os
from src import data_preprocessing as dp
from src import model as m

def main():
    print("=== Air Pollution Analysis & Prediction Pipeline ===")

    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    dp.main()  # This will load, clean, and save the processed CSV

    # Step 2: Train model
    print("\nStep 2: Training model...")
    m.train_model()  # This will train the RandomForest and save model + scaler

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
