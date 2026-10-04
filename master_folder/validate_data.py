import pandas as pd
import sys

# List of expected columns based on the Data Dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"
]

def validate_dataset(file_path):
    try:
        print(f"Validating dataset at: {file_path}")
        df = pd.read_csv(file_path)
        actual_columns = df.columns.tolist()

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
        extra_cols = [col for col in actual_columns if col not in EXPECTED_COLUMNS]

        print("
--- Validation Summary ---")
        print(f"Total Rows: {len(df)}")
        print(f"Total Columns: {len(actual_columns)}")

        if missing_cols:
            print(f"
❌ Error: Missing Expected Columns: {missing_cols}")
            sys.exit(1)
        elif extra_cols:
            print(f"
⚠️ Warning: Found Extra Columns: {extra_cols}")
        else:
            print("
✅ All expected columns are present.")

        print("
--- Data Summary ---")
        print(df.describe(include='all').to_string())
        print("--------------------------")

    except Exception as e:
        print(f"
❌ Error reading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <path_to_csv>")
        sys.exit(1)
    validate_dataset(sys.argv[1])
