import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_data(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Data Cleaning: Remove 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Removed 'Unnamed: 0' column.")

    # Handle missing values (basic imputation for completeness)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    print("Handled missing values.")

    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to {train_path} ({len(train_df)} rows)")
    print(f"Test data saved to {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    input_file = "master_folder/data/tourism.csv"
    out_dir = "master_folder/data"
    os.makedirs(out_dir, exist_ok=True)
    prepare_data(input_file, out_dir)
