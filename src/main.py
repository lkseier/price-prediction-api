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
        print("\n\033[1;34m=== Main Menu ===\033[0m")  # Bold Blue Title
        print("\033[1;34m1. Load dataset and clean\033[0m")             # Bold Blue
        print("\033[1;34m2. Preprocess data for ML\033[0m")             # Bold Blue
        print("\033[1;34m3. Train CatBoost model with Optuna\033[0m")   # Bold Blue
        print("\033[1;31m4. Exit\033[0m")                                # Bold Red

        choice = input("\n\033[1;34mEnter your choice (1-4): \033[0m").strip()

        if choice == "1":
            print("\n\033[1;36m[INFO] Running dataset cleaning...\033[0m")
            cleaner = LoadDatasetAndClean()
            cleaner.run()
            print("\033[1;32m[INFO] Dataset cleaning complete.\033[0m")

        elif choice == "2":
            print("\n\033[1;36m[INFO] Running preprocessing for ML...\033[0m")
            preprocessor = PreprocessorForML()
            preprocessor.run_all()
            print("\033[1;32m[INFO] Preprocessing complete.\033[0m")

        elif choice == "3":
            print("\n\033[1;36m[INFO] Starting training...\033[0m")
            trainer = TrainModel()
            trainer.tune_and_train(n_trials=30)
            print("\033[1;32m[INFO] Training complete.\033[0m")

        elif choice == "4":
            print("\033[1;31mExiting program.\033[0m")
            sys.exit(0)

        else:
            print("\033[1;31mInvalid choice. Please enter 1, 2, 3, or 4.\033[0m")




if __name__ == "__main__":
    main_menu()
