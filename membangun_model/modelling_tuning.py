import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("rf_tuned_m891d5y0695")

train_df = pd.read_csv("train_predictive_maintenance_preprocessed.csv")
test_df  = pd.read_csv("test_predictive_maintenance_preprocessed.csv")

X_train = train_df.drop("Target", axis=1)
y_train = train_df["Target"]

X_test = test_df.drop("Target", axis=1)
y_test = test_df["Target"]

with mlflow.start_run():
    rf_model_tuned = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    
    rf_model_tuned.fit(X_train, y_train)
    
    y_train_pred = rf_model_tuned.predict(X_train)
    y_train_proba = rf_model_tuned.predict_proba(X_train)[:, 1]

    mlflow.log_metric("training_accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("training_f1", f1_score(y_train, y_train_pred))
    mlflow.log_metric("training_log_loss", log_loss(y_train, y_train_proba))
    mlflow.log_metric("training_precision", precision_score(y_train, y_train_pred))
    mlflow.log_metric("training_recall", recall_score(y_train, y_train_pred))
    mlflow.log_metric("training_roc_auc", roc_auc_score(y_train, y_train_proba))
    mlflow.log_metric("training_score", rf_model_tuned.score(X_train, y_train))

    mlflow.log_params(rf_model_tuned.get_params())
    mlflow.sklearn.log_model(rf_model_tuned, "rf_tuned_m891d5y0695")