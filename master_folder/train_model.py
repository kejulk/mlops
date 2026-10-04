import pandas as pd
import os
import joblib
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def main():
    train_path = "master_folder/data/train.csv"
    test_path = "master_folder/data/test.csv"
    model_dir = "master_folder/models"
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load data from workflow artifact (local data folder)
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['ProdTaken', 'CustomerID'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken', 'CustomerID'])
    y_test = test_df['ProdTaken']

    # 2. Preprocessing setup
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # 3. Define a model and pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 4. Parameters for tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.2]
    }

    # 5. MLflow Tracking & Tuning
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Tourism_Package_Prediction")

    print("Starting experiment and tuning...")
    with mlflow.start_run():
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log tuned parameters
        mlflow.log_params(best_params)

        # 6. Evaluate model performance
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        print("
--- Experiment Results ---")
        print("Best Parameters:", best_params)
        print("Accuracy on Test Set:", acc)
        print("
Classification Report:
", classification_report(y_test, y_pred))

        # 7. Save the best model
        model_path = os.path.join(model_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"
Model saved to {model_path}")

if __name__ == "__main__":
    main()
