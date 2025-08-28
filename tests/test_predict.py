import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline


def test_predict_pipeline_on_sample():
    # Use the cleaned dataset and predict the first row
    df = pd.read_csv('notebook/data/cleaned_student_data.csv')
    assert 'G3' in df.columns
    X = df.drop(columns=['G3'])
    sample = X.head(1)
    pp = PredictPipeline()
    preds = pp.predict(sample)
    assert preds is not None
    assert len(preds) == 1

