import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
def handle_missing_values(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("missing", {}).items():
        if col in df.columns:
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode" and not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
    return True, df


def handle_duplicate_rows(df, recommendations):
    df = df.copy()
    if recommendations.get("duplicates", False):
        df = df.drop_duplicates().reset_index(drop=True)
    return True, df


def remove_constant_columns(df, recommendations):
    df = df.copy()
    constants = recommendations.get("constants", [])
    df = df.drop(columns=[c for c in constants if c in df.columns], errors="ignore")
    return True, df


def handle_outliers(df, recommendations):
    df = df.copy()
    for col, bounds in recommendations.get("outliers", {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lower_bound = bounds.get("lower_bound")
            upper_bound = bounds.get("upper_bound")
            if lower_bound is not None and upper_bound is not None:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
    return True, df


def apply_encoding(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("encoding", {}).items():
        if col in df.columns:
            if strategy == "Label Encoding":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            elif strategy == "High Cardinality":
                df = df.drop(columns=[col])
            elif strategy == "One-Hot Encoding":
                ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
                encoded_arr = ohe.fit_transform(df[[col]].astype(str))
                encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df.index)
                df = df.drop(columns=[col]).join(encoded_df)
    return True, df


def apply_scaling(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("scaling", {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "RobustScaler":
                scaler = RobustScaler()
            elif strategy == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
    return True, df
