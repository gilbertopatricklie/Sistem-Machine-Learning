import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_dataset_automated(csv_path):

    df = pd.read_csv(csv_path)

    df.drop(
        columns=["UDI", "Product ID", "Failure Type"],
        inplace=True,
        errors="ignore"
    )

    df["Air temperature [K]"] = df["Air temperature [K]"] - 273.15
    df["Process temperature [K]"] = df["Process temperature [K]"] - 273.15

    df.rename(columns={
    'Air temperature [K]': 'Air temperature [C]',
    'Process temperature [K]': 'Process temperature [C]'
}, inplace=True)

    df = pd.get_dummies(
        df,
        columns=["Type"],
        drop_first=True,
        dtype=int
    )

    target = df["Target"]
    df.drop(columns="Target", inplace=True)
    df["Target"] = target

    X = df.drop(columns="Target")
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled, y_train
    )

    train_df = pd.DataFrame(
        X_train_balanced,
        columns=X_train.columns
    )
    train_df["Target"] = y_train_balanced

    test_df = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )
    test_df["Target"] = y_test.reset_index(drop=True)

    return train_df, test_df, scaler

train_df, test_df, scaler = preprocess_dataset_automated("predictive_maintenance_raw.csv")

train_df.to_csv("train_predictive_maintenance_preprocessed.csv", index=False)
test_df.to_csv("test_predictive_maintenance_preprocessed.csv", index=False)