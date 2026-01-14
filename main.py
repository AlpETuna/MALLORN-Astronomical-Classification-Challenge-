import os
import pandas as pd
from extract_features import extract_features
from train_model import train_model

def main():
    data_dir = "mallorn-astronomical-classification-challenge"
    
    if not os.path.exists("train_features.csv") or not os.path.exists("test_features.csv"):
        print("Feature files not found. Starting feature extraction...")
        extract_features(data_dir)
    else:
        print("Feature files found. Skipping extraction.")
        
    print("\nStarting model training...")
    train_model()
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()

