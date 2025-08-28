import pandas as pd
from src.features.engineer import engineer_features, ENGINEERED_COLS


def test_engineer_features_basic():
    df = pd.DataFrame({
        'failures': [0, 1],
        'studytime': [2, 3],
        'Medu': [2, 4],
        'Fedu': [2, 1],
        'G1': [6, 4],
        'Pstatus': ['T', 'A'],
        'famsize': ['LE3', 'GT3'],
        'address': ['R', 'U'],
        'traveltime': [2, 3],
        'internet': ['no', 'yes'],
        'romantic': ['no', 'yes'],
        'Dalc': [1, 3],
        'Walc': [1, 4],
        'absences': [2, 15],
    })
    out = engineer_features(df)
    # All engineered columns must exist
    for c in ENGINEERED_COLS:
        assert c in out.columns
    # Spot check values
    assert out.loc[0, 'has_failures'] == 0
    assert out.loc[1, 'has_failures'] == 1
    assert out.loc[0, 'high_studytime'] == 0
    assert out.loc[1, 'high_studytime'] == 1
    assert out.loc[0, 'avg_parent_edu'] == 2.0
    assert out.loc[1, 'avg_parent_edu'] == 2.5
    assert out.loc[0, 'early_failure'] == 0
    assert out.loc[1, 'early_failure'] == 1
    assert out.loc[0, 'parents_together'] == 1
    assert out.loc[1, 'parents_together'] == 0
    assert out.loc[0, 'big_family'] == 0
    assert out.loc[1, 'big_family'] == 1
    assert out.loc[0, 'urban_student'] == 0
    assert out.loc[1, 'urban_student'] == 1
    assert out.loc[0, 'long_travel'] == 0
    assert out.loc[1, 'long_travel'] == 1
    assert out.loc[0, 'has_internet'] == 0
    assert out.loc[1, 'has_internet'] == 1
    assert out.loc[0, 'romantic_rel'] == 0
    assert out.loc[1, 'romantic_rel'] == 1
    assert out.loc[0, 'high_alcohol_use'] == 0
    assert out.loc[1, 'high_alcohol_use'] == 1
    assert out.loc[0, 'high_absenteeism'] == 0
    assert out.loc[1, 'high_absenteeism'] == 1

