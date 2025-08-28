import pandas as pd


ENGINEERED_COLS = [
    'has_failures', 'high_studytime', 'avg_parent_edu', 'early_failure',
    'parents_together', 'big_family', 'urban_student', 'long_travel',
    'has_internet', 'romantic_rel', 'high_alcohol_use', 'high_absenteeism'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'failures' in d.columns:
        d['has_failures'] = (d['failures'] > 0).astype(int)
    if 'studytime' in d.columns:
        d['high_studytime'] = (d['studytime'] >= 3).astype(int)
    if {'Medu', 'Fedu'}.issubset(d.columns):
        d['avg_parent_edu'] = (d['Medu'] + d['Fedu']) / 2.0
    if 'G1' in d.columns:
        d['early_failure'] = (d['G1'] < 5).astype(int)
    if 'Pstatus' in d.columns:
        d['parents_together'] = (d['Pstatus'] == 'T').astype(int)
    if 'famsize' in d.columns:
        d['big_family'] = (d['famsize'] == 'GT3').astype(int)
    if 'address' in d.columns:
        d['urban_student'] = (d['address'] == 'U').astype(int)
    if 'traveltime' in d.columns:
        d['long_travel'] = (d['traveltime'] >= 3).astype(int)
    if 'internet' in d.columns:
        d['has_internet'] = (d['internet'] == 'yes').astype(int)
    if 'romantic' in d.columns:
        d['romantic_rel'] = (d['romantic'] == 'yes').astype(int)
    if {'Dalc', 'Walc'}.issubset(d.columns):
        d['high_alcohol_use'] = ((d['Dalc'] + d['Walc']) >= 6).astype(int)
    if 'absences' in d.columns:
        d['high_absenteeism'] = (d['absences'] >= 10).astype(int)
    return d

