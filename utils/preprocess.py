import torch
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():

    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["price"] = housing.target

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).unsqueeze(1)

    y_test = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).unsqueeze(1)

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    ).unsqueeze(1)

    return (
        X_train,
        X_test,
        X_val
        y_train,
        y_test,
        y_val,
        scaler
    )