# Add the project root to the Python path
import sys, os
import json
import catboost

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import os
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import joblib

# === Local imports ===
from src.utils.constants import ML_READY_DATA_FILE, MODELS_DIR, TEST_MODE
from src.utils.train_test_metrics_logger import TrainTestMetricsLogger


MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "catboost_model.pkl")

class TrainModel:
    def __init__(self, data_path=ML_READY_DATA_FILE, target_column="price", model_output_path=MODEL_OUTPUT_PATH, test_size=0.2):
        self.data_path = data_path
        self.target_column = target_column
        self.model_output_path = model_output_path.replace(".pkl", "_test.pkl") if TEST_MODE else model_output_path
        self.test_size = test_size
        self.logger = TrainTestMetricsLogger()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        print(f"[DEBUG] TEST_MODE value = {TEST_MODE}")
        if TEST_MODE:
            print(">>> TEST_MODE is ON: using only 1000 rows")
            X = X.head(1000)
            y = y.head(1000)
        else:
            print(f">>> TEST_MODE is OFF: using full dataset ({len(X)} rows)")

        return train_test_split(X, y, test_size=self.test_size, random_state=42)


    def objective(self, trial, task_type="CPU"):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500 if TEST_MODE else 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "verbose": 0
        }

        model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            task_type=task_type,
            devices="0",
            random_state=42
        )
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, valid_idx in kf.split(self.X_train):
            X_t, X_v = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
            y_t, y_v = self.y_train.iloc[train_idx], self.y_train.iloc[valid_idx]

            model.fit(X_t, y_t, eval_set=(X_v, y_v), use_best_model=True)
            preds = model.predict(X_v)
            rmse = root_mean_squared_error(y_v, preds)
            rmse_scores.append(rmse)

        return sum(rmse_scores) / len(rmse_scores)

    def evaluate_model(self, model, X, y, dataset_name=""):
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = root_mean_squared_error(y, preds)
        print(f"{dataset_name} R²: {r2:.4f}")
        print(f"{dataset_name} MAE: {mae:.2f}")
        print(f"{dataset_name} RMSE: {rmse:.2f}")

    def tune_and_train(self, n_trials=5 if TEST_MODE else 50):
        print("\n\033[1;34m=== CatBoost Training Script ===\033[0m")
        print(f"\033[1;35mMode            : {'TEST' if TEST_MODE else 'FULL'}\033[0m")
        print(catboost.__version__)

        # Try to detect GPU (runtime-safe)
        try:
            test_model = CatBoostRegressor(task_type="GPU", devices="0", verbose=0)
            # Use a dummy dataset with at least 32 samples to avoid "pool has just 2 docs" error
            X_dummy = [[i] for i in range(32)]
            y_dummy = [i for i in range(32)]
            test_model.fit(X_dummy, y_dummy)
            backend = "GPU"
        except Exception as e:
            backend = "CPU"
            print(f"\033[1;31m[WARNING] GPU not available: {e}\033[0m")

        print(f"\033[1;35mBackend         : {backend}\033[0m")
        print(f"\033[1;35mTraining        : Optuna + KFold CV (All Features)\033[0m")   

        # Load data
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

        # Run Optuna tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, task_type=backend), n_trials=n_trials)
        

        # Train best model using GPU
        best_model = CatBoostRegressor(
            **study.best_params,
            loss_function="RMSE",
            random_state=42,
            verbose=0,
            task_type="GPU",
            devices="0"
        )
        best_model.fit(self.X_train, self.y_train)

        # Evaluate model
        print("\nModel Performance:")
        self.evaluate_model(best_model, self.X_train, self.y_train, dataset_name="Train")
        self.evaluate_model(best_model, self.X_test, self.y_test, dataset_name="Test")

        # Save model
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(best_model, self.model_output_path)
        print(f"\nModel saved to: {self.model_output_path}")

        # Save feature list
        features = list(self.X_train.columns)
        features_path = os.path.join(self.model_output_path.replace(".pkl", "_features.json"))
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"Feature list saved to: {features_path}")    

        # Evaluate metrics for logging
        preds_train = best_model.predict(self.X_train)
        preds_test = best_model.predict(self.X_test)

        mae_train = mean_absolute_error(self.y_train, preds_train)
        rmse_train = root_mean_squared_error(self.y_train, preds_train)
        r2_train = r2_score(self.y_train, preds_train)

        mae_test = mean_absolute_error(self.y_test, preds_test)
        rmse_test = root_mean_squared_error(self.y_test, preds_test)
        r2_test = r2_score(self.y_test, preds_test)

        # Log results
        self.logger.log(
            model_name=f"CatBoost CV (All Features){' [TEST]' if TEST_MODE else ''}",
            experiment_name=f"CatBoost Optuna (All Features){' [TEST]' if TEST_MODE else ''}",
            mae_train=mae_train,
            rmse_train=rmse_train,
            r2_train=r2_train,
            mae_test=mae_test,
            rmse_test=rmse_test,
            r2_test=r2_test,
            data_file=self.data_path,
            n_features=self.X_train.shape[1]
        )

        print("\n=== Summary Table ===")
        self.logger.display_table()


if __name__ == "__main__":
    print("\n\033[1;34m=== CatBoost Training Script ===\033[0m")
    print(f"\033[1;35mMode            : {'TEST' if TEST_MODE else 'FULL'}\033[0m")

    # Detect GPU availability with an actual test
    try:
        gpu_test_model = CatBoostRegressor(task_type="GPU", devices="0", verbose=0)
        gpu_test_model.fit([[0], [1]], [0, 1])  # <-- at least 2 distinct values
        backend = "GPU"
    except Exception as e:
        backend = "CPU"
        print(f"\033[1;31mGPU not available: {e}\033[0m")

    print(f"\033[1;35mBackend         : {backend}\033[0m")
    print(f"\033[1;35mTraining        : Optuna + KFold CV (All Features)\033[0m")

    # Launch training
    trainer = TrainModel()
    trainer.tune_and_train()