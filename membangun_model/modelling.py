import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("rf_model")
mlflow.sklearn.autolog()

train_df = pd.read_csv('train_predictive_maintenance_preprocessed.csv')
test_df  = pd.read_csv('test_predictive_maintenance_preprocessed.csv')

X_train = train_df.drop('Target', axis=1)
y_train = train_df['Target']

X_test = test_df.drop('Target', axis=1)
y_test = test_df['Target']

with mlflow.start_run():
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    