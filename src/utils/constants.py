import os

# === Base directory (1 level up from current file) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# === Subdirectories ===
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
ML_READY_DIR = os.path.join(DATA_DIR, "ml_ready")
CLEANED_FOR_ML_DATA_DIR = os.path.join(DATA_DIR, "ml_ready")  
MODEL_OUTPUT_DIR = os.path.join(DATA_DIR, "ML")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_DIR = os.path.join(BASE_DIR, "database")  
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")


# === Data files ===
RAW_DATA_FILE = os.path.join(RAW_DIR, "immoweb_real_estate.csv")
CLEANED_DATA_FILE = os.path.join(CLEANED_DIR, "immoweb_real_estate_cleaned_dataset.csv")
ML_READY_DATA_FILE = os.path.join(DATA_DIR, "immoweb_real_estate_ml_ready.csv")

ML_READY_DATA_FILE = os.path.join(ML_READY_DIR, "immoweb_real_estate_ml_ready.csv")
ML_READY_SAMPLE_XLSX = os.path.join(ML_READY_DIR, "immoweb_real_estate_ml_ready_sample10.xlsx")

# === Dev mode ===
TEST_MODE = True


