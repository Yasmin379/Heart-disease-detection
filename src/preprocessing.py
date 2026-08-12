import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def split_features_target(df):
    X = df.drop(columns=["heart_disease"])
    y = df["heart_disease"]

    return X, y


def split_train_test(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

def preprocess_data():
    path = "data/heart_disease.csv"
    df = load_data(path)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return X_train, X_test, y_train, y_test