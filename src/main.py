import sys, os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Imports from local modules
from src.load_dataset_and_clean import LoadDatasetAndClean
from src.preprocess_for_ml import PreprocessorForML
from src.train_model import TrainModel


def main_menu():
    while True:
        print("\nMain Menu ---")
        print("1. Load dataset and clean")
        print("2. Preprocess data for ML")
        print("3. Train CatBoost model with Optuna")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print("\n[INFO] Running dataset cleaning...")
            cleaner = LoadDatasetAndClean()
            cleaner.run()
            print("[INFO] Dataset cleaning complete.")

        elif choice == "2":
            print("\n[INFO] Running preprocessing for ML...")
            preprocessor = PreprocessorForML()
            preprocessor.run_all()
            print("[INFO] Preprocessing complete.")

        elif choice == "3":
            print("\n[INFO] Starting training...")
            trainer = TrainModel(
                data_path="data/ml_ready/immoweb_real_estate_ml_ready.csv",
                target_column="price",
                model_output_path="models/catboost_model.pkl"
            )
            trainer.tune_and_train(n_trials=30)
            print("[INFO] Training complete.")

        elif choice == "4":
            print("Exiting program.")
            sys.exit(0)

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main_menu()
