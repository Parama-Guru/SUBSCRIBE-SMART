import os
import joblib
import logging
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(filename="ml_pipeline.log", level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")

# Models and hyperparameters
models = {
    "Logistic Regression": (LogisticRegression(), {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "saga"]
    }),
    "Decision Tree": (DecisionTreeClassifier(), {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2'],
    }),
    "SVM": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }),
    "XGBoost": (XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 10],
        "n_estimators": [50, 100, 200],
        'colsample_bytree': [0.3, 0.5, 0.7, 1],
        'subsample': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_lambda': [1, 1.5, 2],
    })
}

# Load preprocessed datasets
try:
    X_train = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/X_train.csv').values
    X_val = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/X_val.csv').values
    X_test = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/X_test.csv').values
    y_train = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/y_train.csv').values.ravel()
    y_val = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/y_val.csv').values.ravel()
    y_test = pd.read_csv('/Users/guru/vsc/ml/myself/sckit/preprocessed_dataset/y_test.csv').values.ravel()
    logging.info("Datasets loaded successfully.")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    raise

def train_and_evaluate_models(X_train, X_val, y_train, y_val, models):
    best_model = None
    best_score = 0
    performance_report = {}

    for model_name, (model, params) in models.items():
        logging.info(f"Training {model_name} with Grid Search")

        try:
            # Debugging shapes and model details
            logging.debug(f"Model: {model_name}, Params: {params}")
            logging.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            # Configure GridSearchCV with cross-validation
            grid_search = GridSearchCV(model, params, scoring='accuracy', cv=3, n_jobs=-1, refit=True)
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='binary') if len(set(y_train)) == 2 else precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='binary') if len(set(y_train)) == 2 else recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='binary') if len(set(y_train)) == 2 else f1_score(y_val, y_pred, average='weighted')

            # Log each metric for traceability
            logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            performance_report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_params": grid_search.best_params_
            }

            # Update best model if accuracy improves
            if accuracy > best_score:
                best_score = accuracy
                best_model = best_estimator
                
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            continue

    # Save the best model in .joblib format
    if best_model:
        joblib.dump(best_model, "best_model.joblib")
        logging.info("Best model saved as 'best_model.joblib'.")
    else:
        logging.warning("No model was successfully trained.")

    # Write performance report to text file
    with open("performance_report.txt", "w") as report_file:
        if performance_report:
            for model, metrics in performance_report.items():
                report_file.write(f"{model}:\n{json.dumps(metrics, indent=4)}\n\n")
            logging.info("Model performance report saved as 'performance_report.txt'.")
        else:
            report_file.write("No model was successfully trained.\n")
            logging.warning("Performance report is empty.")

if __name__ == "__main__":
    train_and_evaluate_models(X_train, X_val, y_train, y_val, models)