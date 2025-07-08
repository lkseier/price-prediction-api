import sys, os
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src.data_cleaner import DataCleaner
from src.utils.constants import RAW_DATA_FILE, CLEANED_DATA_FILE, ML_READY_DATA_FILE

class LoadDatasetAndClean:
    def __init__(self, missing_threshold=0.5, row_threshold=0.7):
        self.missing_threshold = missing_threshold
        self.row_threshold = row_threshold
        self.df_raw = None
        self.df_cleaned = None

    def load_raw_data(self):
        print(f"Loading dataset from: {RAW_DATA_FILE}")
        self.df_raw = pd.read_csv(RAW_DATA_FILE)
        print(f"Initial shape: {self.df_raw.shape}")

    def clean_data(self):
        cleaner = DataCleaner(
            missing_threshold=self.missing_threshold,
            row_threshold=self.row_threshold
        )

        # Step 1: Drop columns with too many missing values
        df_step1 = cleaner.drop_columns_with_missing_values(self.df_raw)

        # Step 2: Drop rows with too many missing values
        df_step2 = cleaner.drop_rows_with_missing_values(df_step1)

        # Step 3: Remove outliers
        self.df_cleaned = cleaner.remove_outliers(df_step2)

    def save_outputs(self):
        # Save full cleaned dataset
        os.makedirs(os.path.dirname(CLEANED_DATA_FILE), exist_ok=True)
        self.df_cleaned.to_csv(CLEANED_DATA_FILE, index=False)
        print(f"Cleaned dataset saved to: {CLEANED_DATA_FILE}")

        # Save sample for review
        os.makedirs(os.path.dirname(ML_READY_DATA_FILE), exist_ok=True)
        excel_sample_path = ML_READY_DATA_FILE.replace(".csv", "_sample10.xlsx")
        self.df_cleaned.head(10).to_excel(excel_sample_path, index=False)
        print(f"Sample Excel file saved to: {excel_sample_path}")

    def run(self):
        self.load_raw_data()
        self.clean_data()
        self.save_outputs()
