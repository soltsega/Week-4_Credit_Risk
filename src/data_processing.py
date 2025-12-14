import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class AggregatorTransformer(BaseEstimator, TransformerMixin):
    """Aggregates transaction-level data into per-customer features.

    The transformer expects a transaction-level DataFrame with at least
    the following columns: 'CustomerId', 'TransactionDate', 'Value', 'Amount',
    'ChannelId', 'ProductCategory', 'ProviderId'.
    """

    def __init__(self, id_col: str = "CustomerId", snapshot_date: Optional[pd.Timestamp] = None):
        self.id_col = id_col
        self.snapshot_date = snapshot_date

    def fit(self, X: pd.DataFrame, y=None):
        if self.snapshot_date is None:
            self.snapshot_date_ = X["TransactionDate"].max()
        else:
            self.snapshot_date_ = pd.to_datetime(self.snapshot_date)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "snapshot_date_"):
            self.fit(X)

        df = X.copy()
        # Ensure TransactionDate is datetime
        if not np.issubdtype(df["TransactionDate"].dtype, np.datetime64):
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])  # type: ignore

        group = df.groupby(self.id_col)

        agg = group.agg(
            recency=("TransactionDate", lambda x: (self.snapshot_date_ - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary_total=("Value", "sum"),
            monetary_mean=("Value", "mean"),
            amount_sum=("Amount", "sum"),
            amount_mean=("Amount", "mean"),
            amount_std=("Amount", "std"),
            amount_min=("Amount", "min"),
            amount_max=("Amount", "max"),
            unique_products=("ProductId", pd.Series.nunique),
            unique_product_categories=("ProductCategory", pd.Series.nunique),
            unique_channels=("ChannelId", pd.Series.nunique),
            unique_providers=("ProviderId", pd.Series.nunique),
        )

        # Additional features based on group
        # percent refunds (Amount < 0)
        refunds = group["Amount"].apply(lambda s: (s < 0).sum())
        agg["refund_count"] = refunds
        agg["refund_ratio"] = agg["refund_count"] / agg["frequency"]

        # top channel proportion
        top_channel = group["ChannelId"].apply(lambda s: s.value_counts().iloc[0])
        agg["top_channel_count"] = top_channel
        agg["top_channel_prop"] = agg["top_channel_count"] / agg["frequency"]

        # mode transaction hour
        try:
            top_hour = group["TransactionHour"].apply(lambda s: s.mode().iloc[0])
            agg["top_transaction_hour"] = top_hour
        except Exception:
            agg["top_transaction_hour"] = group["TransactionHour"].apply(lambda s: s.iloc[0])

        # Fill NaN in std with 0 for single-transaction customers
        agg["amount_std"] = agg["amount_std"].fillna(0.0)

        # Reset index to have CustomerId column
        agg = agg.reset_index()

        # Drop counters used for ratio derivations (keep them if desired)
        agg = agg.drop(columns=["refund_count", "top_channel_count"], errors="ignore")

        return agg


class PreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Imputes and scales numeric features and optionally encodes categorical ones.

    Input: DataFrame output from AggregatorTransformer.
    Output: DataFrame with numeric features scaled and categorical features encoded.
    """

    def __init__(self, numeric_impute_strategy: str = "median", scaler: Optional[StandardScaler] = None):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.scaler = scaler or StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # Identify numeric columns (exclude the id column)
        self.id_col = "CustomerId" if "CustomerId" in X.columns else None
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.id_col and self.id_col in self.numeric_cols:
            self.numeric_cols.remove(self.id_col)

        self.imputer_ = SimpleImputer(strategy=self.numeric_impute_strategy)
        self.imputer_.fit(X[self.numeric_cols])
        # Fit scaler on imputed numeric columns
        X_num = pd.DataFrame(self.imputer_.transform(X[self.numeric_cols]), columns=self.numeric_cols)
        self.scaler.fit(X_num)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Impute and scale numeric
        if len(self.numeric_cols) > 0:
            X_num = pd.DataFrame(self.imputer_.transform(X[self.numeric_cols]), columns=self.numeric_cols, index=X.index)
            X_num = pd.DataFrame(self.scaler.transform(X_num), columns=self.numeric_cols, index=X.index)
            X[self.numeric_cols] = X_num
        return X


def build_feature_pipeline(snapshot_date: Optional[pd.Timestamp] = None) -> Pipeline:
    """Builds the feature engineering pipeline.

    Returns an sklearn Pipeline that when called with a transaction-level DataFrame
    will return a per-customer feature DataFrame with numeric columns scaled.
    """
    aggregator = AggregatorTransformer(snapshot_date=snapshot_date)
    preprocessor = PreprocessorTransformer()
    pipeline = Pipeline([
        ("aggregator", aggregator),
        ("preprocessor", preprocessor),
    ])
    return pipeline


def compute_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute RFM metrics (recency, frequency, monetary) per CustomerId."""
    df = df.copy()
    if snapshot_date is None:
        snapshot_date = df["TransactionDate"].max()
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Value", "sum"),
    ).reset_index()
    return rfm


if __name__ == "__main__":
    # Quick demo when run standalone
    df = pd.read_csv("data/processed/processed_data.csv", parse_dates=["TransactionStartTime", "TransactionDate"])  # type: ignore
    print("Building pipeline and transforming sample data...")
    pl = build_feature_pipeline()
    features = pl.fit_transform(df)
    print(features.head().to_string())
