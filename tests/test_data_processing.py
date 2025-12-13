import pandas as pd
import numpy as np
from src.data_processing import compute_rfm, build_feature_pipeline


def make_sample_transactions():
    data = {
        'TransactionId': ['t1', 't2', 't3', 't4'],
        'CustomerId': ['c1', 'c1', 'c2', 'c3'],
        'Value': [100, 200, 50, 500],
        'Amount': [100.0, -50.0, 50.0, 500.0],
        'TransactionDate': pd.to_datetime(['2019-02-10','2019-02-11','2019-01-01','2019-02-13']),
        'TransactionHour': [10, 11, 12, 13],
        'ChannelId': ['ch1', 'ch2', 'ch1', 'ch3'],
        'ProductId': ['p1', 'p2', 'p1', 'p3'],
        'ProductCategory': ['airtime', 'airtime', 'utility', 'financial_services'],
        'ProviderId': ['prov1', 'prov1', 'prov2', 'prov3'],
    }
    return pd.DataFrame(data)


def test_compute_rfm_recency_frequency_monetary():
    df = make_sample_transactions()
    # use snapshot date 2019-02-13
    rfm = compute_rfm(df, snapshot_date='2019-02-13')
    # check customer counts
    assert rfm.shape[0] == 3
    # c1 has two transactions; frequency should be 2
    f_c1 = rfm.loc[rfm['CustomerId'] == 'c1', 'frequency'].iloc[0]
    assert f_c1 == 2
    # recency for c2 should be days between 2019-02-13 and 2019-01-01 = 43 days
    r_c2 = rfm.loc[rfm['CustomerId'] == 'c2', 'recency'].iloc[0]
    assert r_c2 == 43


def test_pipeline_aggregates_features():
    df = make_sample_transactions()
    pipeline = build_feature_pipeline(snapshot_date='2019-02-13')
    features = pipeline.fit_transform(df)
    # we expect one row per unique CustomerId (3 customers)
    assert features.shape[0] == 3
    # expected columns include recency, frequency, monetary_total
    for col in ['recency', 'frequency', 'monetary_total', 'amount_sum', 'unique_channels']:
        assert col in features.columns
