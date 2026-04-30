"""Data processing nodes – clean, encode, split and scale."""

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel",
}

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "Age":         (10, 70),
    "SystolicBP":  (60, 200),
    "DiastolicBP": (40, 140),
    "BS":          (3.0, 20.0),
    "BodyTemp":    (95.0, 106.0),
    "HeartRate":   (40, 160),
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw maternal-health data."""
    logger.info("Raw shape: %s", df.shape)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df.copy()
    df["RiskLevel"] = df["RiskLevel"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows.", before - len(df))

    before = len(df)
    df = df.dropna()
    logger.info("Dropped %d rows with nulls.", before - len(df))

    before = len(df)
    for col, (lo, hi) in VALID_RANGES.items():
        df = df[df[col].between(lo, hi)]
    logger.info(
        "Dropped %d rows with out-of-range values. Clean shape: %s",
        before - len(df),
        df.shape,
    )

    return df.reset_index(drop=True)


def encode_target(df: pd.DataFrame, risk_mapping: Dict[str, int]) -> pd.DataFrame:
    """Encode the RiskLevel target column using mapping from parameters.yml."""
    df = df.copy()
    unknown = set(df["RiskLevel"].unique()) - set(risk_mapping.keys())
    if unknown:
        raise ValueError(f"Unknown RiskLevel labels: {unknown}")

    df["RiskLevel"] = df["RiskLevel"].map(risk_mapping)
    logger.info(
        "Target distribution after encoding:\n%s",
        df["RiskLevel"].value_counts().to_string(),
    )
    return df


def split_and_scale(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """
    Stratified train/test split followed by MinMax scaling.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Split sizes → train: %d  test: %d", len(X_train), len(X_test))

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler