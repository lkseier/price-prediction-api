import sys, os
import pandas as pd
import shutil

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.constants import CLEANED_DATA_FILE, CLEANED_DIR, ML_READY_DIR, ML_READY_DATA_FILE, ML_READY_SAMPLE_XLSX
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.preprocessing_pipeline import PreprocessingPipeline


class PreprocessorForML:
    def __init__(self):
        # Fix: provide correct path to DataLoader
        self.loader = DataLoader(CLEANED_DATA_FILE)
        self.df = None
        self.df_encoded = None

    def load_and_clean_data(self):
        print("[INFO] Loading dataset...")
        self.df = self.loader.load_data()
        self.df = self.loader.clean_booleans(self.df, bool_cols=["hasLivingRoom", "hasTerrace"])
        self.df = self.loader.drop_columns(self.df, columns_to_drop=["Unnamed: 0", "id", "url"])
        self.df = self.loader.drop_na_targets(self.df, target_col="price")

        # Derived feature
        self.df["building_age"] = 2025 - self.df["buildingConstructionYear"]
        print("[INFO] Data loaded and cleaned. Shape:", self.df.shape)

    def apply_pipeline(self):
        print("[INFO] Applying preprocessing pipeline...")
        pipeline = PreprocessingPipeline(
            df=self.df,
            target_col="price",
            drop_cols=["price_per_m2", "log_price"],
        )
        self.df_encoded = pipeline.fit_transform()

        # Check for unwanted columns
        for col in ["price_per_m2", "log_price"]:
            if col in self.df_encoded.columns:
                print(f"[WARNING] Unwanted column still present: {col}")
            else:
                print(f"[INFO] Column removed: {col}")

    def save_outputs(self):
        print("[INFO] Saving preprocessed dataset...")
        if os.path.exists(ML_READY_DIR):
            shutil.rmtree(ML_READY_DIR)
        os.makedirs(ML_READY_DIR, exist_ok=True)

        self.df_encoded.to_csv(ML_READY_DATA_FILE, index=False)
        self.df_encoded.head(10).to_excel(ML_READY_SAMPLE_XLSX, index=False)

        print(f"[SUCCESS] Dataset ready. Shape: {self.df_encoded.shape}")
        print(f"[SAVED] CSV: {ML_READY_DATA_FILE}")
        print(f"[SAVED] Sample Excel: {ML_READY_SAMPLE_XLSX}")

    def run_all(self):
        self.load_and_clean_data()
        self.apply_pipeline()
        self.save_outputs()
